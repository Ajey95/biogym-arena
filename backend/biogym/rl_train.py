"""Reward-driven LLM adapter training with audited BioGym tool episodes.

Run from a GPU Linux runtime: python -m biogym.rl_train --output /content/biogym-run
This module imports TRL only when invoked, leaving the API server lightweight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .rl_suite import (BioGymRLToolEnv, SCHEMA_VERSION, SCORER_VERSION, SUITE_HASH,
                       reward_outcome, reward_process, split_tasks)


DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def _file_hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _summarize_traces(directory: Path) -> dict:
    rows = []
    for path in sorted(directory.glob("*.json")):
        trace = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"case_id": trace["case_id"], "attempt_id": trace["attempt_id"],
                     "success": trace["success"], "done": trace["done"],
                     "reward": trace["reward"],
                     "labops": trace["labops"], "researchsense": trace["researchsense"]})
    return {"attempts": len(rows), "terminal_attempts": sum(row["done"] for row in rows),
            "successes": sum(row["success"] for row in rows),
            "mean_reward": sum(row["reward"] for row in rows) / len(rows) if rows else None,
            "rows": rows}


def validate_warm_adapter(adapter_dir: Path, model_id: str) -> dict:
    """Reject stale, tampered or held-out-contaminated supervised adapters."""
    manifest_path = adapter_dir.parent / "warmstart_manifest.json"
    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report["model_id"] != model_id:
        raise ValueError("warm adapter model differs from requested base model")
    if report["suite_hash"] != SUITE_HASH:
        raise ValueError("warm adapter suite differs from current tasks")
    if report["optimizer_steps"] < 1:
        raise ValueError("warm adapter has no optimizer updates")
    train_ids = {task.id for task in split_tasks("train")}
    if not report["case_ids"] or set(report["case_ids"]) - train_ids:
        raise ValueError("warm adapter inventory includes a non-train case")
    weights = adapter_dir / "adapter_model.safetensors"
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != report["adapter_hashes"][weights.name]:
        raise ValueError("warm adapter tensor hash mismatch")
    demonstrations = adapter_dir.parent / "train_demonstrations.jsonl"
    if hashlib.sha256(demonstrations.read_bytes()).hexdigest() != report["demonstration_sha256"]:
        raise ValueError("warm adapter demonstration hash mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--warm-adapter", type=Path)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("max-steps must be positive")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this LLM training command")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is not empty; choose a fresh directory for each run")
    output.mkdir(parents=True, exist_ok=True)

    train_tasks = split_tasks("train")
    test_tasks = split_tasks("test")
    if not train_tasks or not test_tasks:
        raise RuntimeError("train/test cases are missing")
    if {task.source_key for task in train_tasks} & {task.source_key for task in test_tasks}:
        raise RuntimeError("source entity leakage between train and test")

    def rows(tasks):
        return {"prompt": [[{"role": "user", "content":
                             "Complete the BioGym scientific task in the environment. "
                             "Use tool evidence for your final decision."}]
                            for _ in tasks],
                "case_id": [task.id for task in tasks],
                "split": [task.split for task in tasks]}

    train_dataset = Dataset.from_dict(rows(train_tasks))
    test_dataset = Dataset.from_dict(rows(test_tasks))
    gpu = torch.cuda.get_device_properties(0)
    if gpu.total_memory < 14_000_000_000:
        raise RuntimeError("This recipe requires at least 14 GB GPU memory")
    warm_report = None
    model_for_training = args.model
    if args.warm_adapter:
        adapter_dir = args.warm_adapter.resolve()
        warm_report = validate_warm_adapter(adapter_dir, args.model)
        warm_archive = output / "warmstart"
        warm_archive.mkdir()
        shutil.copy2(adapter_dir.parent / "warmstart_manifest.json", warm_archive)
        shutil.copy2(adapter_dir.parent / "train_demonstrations.jsonl", warm_archive)
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
        model_for_training = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    config = GRPOConfig(
        output_dir=str(output / "trainer"),
        model_init_kwargs=None if args.warm_adapter else {"dtype": "float16"},
        fp16=True, bf16=False, use_vllm=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        per_device_eval_batch_size=2,
        num_generations=2, num_generations_eval=2,
        max_completion_length=1024,
        max_tool_calling_iterations=6,
        chat_template_kwargs={"enable_thinking": False},
        learning_rate=2e-5, lr_scheduler_type="constant",
        max_steps=args.max_steps, logging_steps=1, save_steps=max(1, args.max_steps // 2),
        save_total_limit=2, gradient_checkpointing=True,
        report_to="none", log_completions=True,
        seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False,
    )
    if config.per_device_eval_batch_size != config.generation_batch_size:
        raise RuntimeError("evaluation batch size must match the stateful environment pool")
    peft = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    trainer = GRPOTrainer(
        model=model_for_training, args=config, train_dataset=train_dataset,
        eval_dataset=test_dataset, reward_funcs=[reward_outcome, reward_process],
        peft_config=None if args.warm_adapter else peft,
        environment_factory=BioGymRLToolEnv,
    )
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if trainable < 1:
        raise RuntimeError("no trainable LLM parameters")
    pre = output / "pre_adapter"
    post = output / "post_adapter"
    trainer.model.save_pretrained(pre)
    trainer.processing_class.save_pretrained(pre)
    pre_hashes = _file_hashes(pre)

    train_dir = output / "traces" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    os.environ["BIOGYM_RL_TRACE_DIR"] = str(train_dir)
    try:
        training_result = trainer.train()
    finally:
        os.environ.pop("BIOGYM_RL_TRACE_DIR", None)
    trainer.save_state()
    trainer.model.save_pretrained(post)
    trainer.processing_class.save_pretrained(post)
    post_hashes = _file_hashes(post)
    if pre_hashes.get("adapter_model.safetensors") == post_hashes.get("adapter_model.safetensors"):
        raise RuntimeError("adapter tensor hash did not change")
    if trainer.state.global_step < 1:
        raise RuntimeError("optimizer did not step")

    # Reopen the saved adapter through the standard PEFT loader and perform a
    # short generation. A changed file hash alone is not a usable checkpoint.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    reloaded_base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32,
                                                         device_map="cpu")
    reloaded = PeftModel.from_pretrained(reloaded_base, post).eval()
    reloaded_tokenizer = AutoTokenizer.from_pretrained(post)
    probe = reloaded_tokenizer("BioGym evidence:", return_tensors="pt")
    with torch.no_grad():
        generated = reloaded.generate(**probe, max_new_tokens=3, do_sample=False)
    checkpoint_probe = reloaded_tokenizer.decode(generated[0], skip_special_tokens=True)
    if not checkpoint_probe:
        raise RuntimeError("reloaded checkpoint produced no text")

    train_summary = _summarize_traces(train_dir)
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None

    report = {
        "schema_version": SCHEMA_VERSION, "scorer_version": SCORER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite_hash": SUITE_HASH, "model_id": args.model,
        "code_commit": git_commit, "seed": args.seed,
        "warmstart_manifest": "warmstart/warmstart_manifest.json" if warm_report else None,
        "warmstart_demonstration_sha256": (warm_report["demonstration_sha256"]
                                            if warm_report else None),
        "warmstart_optimizer_steps": warm_report["optimizer_steps"] if warm_report else None,
        "model_revision": getattr(trainer.model.config, "_commit_hash", None),
        "train_entities": len({task.source_key for task in train_tasks}),
        "test_entities": len({task.source_key for task in test_tasks}),
        "train_case_count": len(train_tasks), "test_case_count": len(test_tasks),
        "optimizer_steps": trainer.state.global_step, "trainable_parameters": trainable,
        "checkpoint_reload_probe": checkpoint_probe,
        "pre_adapter_hashes": pre_hashes, "post_adapter_hashes": post_hashes,
        "gpu": gpu.name, "gpu_memory_bytes": gpu.total_memory,
        "torch": torch.__version__, "python": platform.python_version(),
        "trl": __import__("trl").__version__,
        "training_loss": training_result.training_loss,
        "train_trace_summary": train_summary,
        "evaluation_status": "requires_fresh_process_pre_post_and_transfer_runs",
        "limitations": [
            "This trains a Qwen LLM adapter, not Sarvam 30B.",
            "Source data are cached UniProt/RCSB metadata and authored tasks, not biological validation.",
            "This small pilot does not establish general scientific-agent reliability.",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    print(json.dumps({key: report[key] for key in (
        "model_id", "optimizer_steps", "trainable_parameters", "train_trace_summary")}, indent=2))


if __name__ == "__main__":
    main()
