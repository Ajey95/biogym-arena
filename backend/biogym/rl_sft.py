"""Train a tool-use LoRA warm start from verified BioGym train trajectories.

Run on a GPU Linux runtime: python -m biogym.rl_sft --output /content/biogym-sft
The resulting adapter can initialize GRPO with rl_train --warm-adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .rl_demonstrations import build_demonstration
from .rl_suite import BioGymRLToolEnv, SCHEMA_VERSION, SUITE_HASH, split_tasks
from .rl_train import DEFAULT_MODEL, _file_hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("warm-start output must be a fresh directory")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers.utils import get_json_schema
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the LLM warm start")
    output.mkdir(parents=True, exist_ok=True)
    environment = BioGymRLToolEnv()
    methods = [getattr(environment, name) for name, _ in
               inspect.getmembers(type(environment), predicate=inspect.isfunction)
               if not name.startswith("_") and name != "reset"]
    schemas = [get_json_schema(method) for method in methods]
    demonstrations = [build_demonstration(task) for task in split_tasks("train")]
    if len(demonstrations) != 80 or any(item["suite_hash"] != SUITE_HASH for item in demonstrations):
        raise RuntimeError("train-only demonstration inventory differs from suite")
    training_file = output / "train_demonstrations.jsonl"
    training_file.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in demonstrations),
        encoding="utf-8",
    )
    dataset = Dataset.from_list(
        [{"messages": item["messages"], "tools": schemas} for item in demonstrations],
        on_mixed_types="use_json",
    )
    config = SFTConfig(
        output_dir=str(output / "trainer"),
        model_init_kwargs={"dtype": "float16"},
        fp16=True, bf16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=args.epochs, learning_rate=5e-5,
        max_length=4096, assistant_only_loss=True,
        gradient_checkpointing=True, packing=False,
        logging_steps=5, save_strategy="no", report_to="none",
        seed=args.seed, data_seed=args.seed,
    )
    peft = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=dataset,
                         peft_config=peft)
    tokenizer = trainer.processing_class
    lengths = [len(tokenizer.apply_chat_template(
        item["messages"], tools=schemas, tokenize=True,
        add_generation_prompt=False,
    )) for item in demonstrations]
    if max(lengths) > config.max_length:
        raise RuntimeError(f"demonstration exceeds SFT context: max {max(lengths)} tokens")
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    if trainable < 1:
        raise RuntimeError("no trainable adapter parameters")
    result = trainer.train()
    if trainer.state.global_step < 1:
        raise RuntimeError("warm-start optimizer did not step")
    adapter_dir = output / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    adapter_hashes = _file_hashes(adapter_dir)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    report = {
        "schema_version": SCHEMA_VERSION, "suite_hash": SUITE_HASH,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model, "code_commit": commit, "seed": args.seed,
        "training_cases": len(demonstrations),
        "case_ids": [item["case_id"] for item in demonstrations],
        "demonstration_sha256": hashlib.sha256(training_file.read_bytes()).hexdigest(),
        "tool_schema_sha256": hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest(),
        "max_demonstration_tokens": max(lengths),
        "optimizer_steps": trainer.state.global_step,
        "trainable_parameters": trainable, "training_loss": result.training_loss,
        "adapter_hashes": adapter_hashes,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "trl": __import__("trl").__version__, "python": platform.python_version(),
        "limitations": "Verifier-generated train trajectories, not expert-reviewed biological evidence.",
    }
    (output / "warmstart_manifest.json").write_text(
        json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("model_id", "training_cases", "optimizer_steps",
                       "max_demonstration_tokens", "trainable_parameters")}, indent=2))


if __name__ == "__main__":
    main()
