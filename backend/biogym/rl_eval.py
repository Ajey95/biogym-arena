"""Evaluate one frozen BioGym LLM adapter in a fresh process on held-out cases.

Run pre and post adapters separately to give both the same GPU memory budget.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from .rl_suite import BioGymRLToolEnv, SCHEMA_VERSION, SCORER_VERSION, SUITE_HASH
from .rl_suite import reward_outcome, reward_process, split_tasks
from .rl_train import _summarize_traces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adapter", choices=("pre", "post"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("evaluation output must be a fresh directory")
    training_report = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if training_report["suite_hash"] != SUITE_HASH:
        raise ValueError("evaluation suite differs from training run")
    adapter_dir = run_dir / f"{args.adapter}_adapter"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    if adapter_hash != training_report[f"{args.adapter}_adapter_hashes"][adapter_path.name]:
        raise ValueError("adapter hash differs from training manifest")

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    output.mkdir(parents=True, exist_ok=True)
    test_tasks = split_tasks("test")
    train_tasks = split_tasks("train")

    def rows(tasks):
        return Dataset.from_dict({
            "prompt": [[{"role": "user", "content":
                         "Complete the BioGym scientific task in the environment. "
                         "Use tool evidence for your final decision."}]
                        for _ in tasks],
            "case_id": [task.id for task in tasks],
            "split": [task.split for task in tasks],
        })

    config = GRPOConfig(
        output_dir=str(output / "trainer"),
        fp16=True, bf16=False, use_vllm=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        per_device_eval_batch_size=2,
        num_generations=2, num_generations_eval=2,
        max_completion_length=1024, max_tool_calling_iterations=6,
        chat_template_kwargs={"enable_thinking": False},
        max_steps=1, report_to="none", log_completions=True,
        seed=training_report["seed"], data_seed=training_report["seed"],
        remove_unused_columns=False,
    )
    base = AutoModelForCausalLM.from_pretrained(training_report["model_id"], dtype=torch.float16)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, args=config,
        train_dataset=rows(train_tasks), eval_dataset=rows(test_tasks),
        reward_funcs=[reward_outcome, reward_process],
        environment_factory=BioGymRLToolEnv,
    )
    trace_dir = output / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BIOGYM_RL_TRACE_DIR"] = str(trace_dir)
    try:
        set_seed(training_report["seed"] + 10000)
        metrics = trainer.evaluate(metric_key_prefix="heldout")
    finally:
        os.environ.pop("BIOGYM_RL_TRACE_DIR", None)
    summary = _summarize_traces(trace_dir)
    expected = {task.id for task in test_tasks}
    if {row["case_id"] for row in summary["rows"]} != expected:
        raise RuntimeError("held-out evaluation did not cover every case")
    if summary["attempts"] != len(test_tasks) * config.num_generations_eval:
        raise RuntimeError("held-out evaluation has unexpected attempt count")
    report = {
        "schema_version": SCHEMA_VERSION, "scorer_version": SCORER_VERSION,
        "suite_hash": SUITE_HASH, "created_at": datetime.now(timezone.utc).isoformat(),
        "training_run": str(run_dir), "training_commit": training_report["code_commit"],
        "model_id": training_report["model_id"], "adapter": args.adapter,
        "adapter_sha256": adapter_hash, "seed": training_report["seed"],
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "trl": __import__("trl").__version__, "python": platform.python_version(),
        "metrics": metrics, "heldout": summary,
    }
    (output / "evaluation_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"adapter": args.adapter, "attempts": summary["attempts"],
                      "successes": summary["successes"], "mean_reward": summary["mean_reward"]}))


if __name__ == "__main__":
    main()
