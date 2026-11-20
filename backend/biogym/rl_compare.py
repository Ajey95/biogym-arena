"""Validate two isolated held-out evaluations and bundle the complete pilot run."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .rl_suite import SUITE_HASH, split_tasks


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_case(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    return dict(grouped)


def _mean(rows: list[dict], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def _zip_tree(bundle: zipfile.ZipFile, source: Path, prefix: str) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            bundle.write(path, f"{prefix}/{path.relative_to(source).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pre-eval", type=Path, required=True)
    parser.add_argument("--post-eval", type=Path, required=True)
    parser.add_argument("--pre-log", type=Path, required=True)
    parser.add_argument("--post-log", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--evaluation-code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("comparison bundle already exists")
    run = _read(args.run_dir / "run_manifest.json")
    if run["suite_hash"] != SUITE_HASH:
        raise ValueError("training run belongs to a different suite version")
    pre = _read(args.pre_eval / "evaluation_manifest.json")
    post = _read(args.post_eval / "evaluation_manifest.json")
    evaluation_split = pre.get("split", "test")
    if evaluation_split not in {"test", "transfer"} or post.get("split", "test") != evaluation_split:
        raise ValueError("evaluation splits differ or are not held-out")
    expected_case_ids = {task.id for task in split_tasks(evaluation_split)}
    generations = pre.get("generations_per_case", 2)
    if post.get("generations_per_case", 2) != generations or generations < 1:
        raise ValueError("evaluation attempt budgets differ")
    expected_attempts = len(expected_case_ids) * generations
    for label, item in (("pre", pre), ("post", post)):
        if item["adapter"] != label or item["suite_hash"] != run["suite_hash"]:
            raise ValueError(f"{label} evaluation has wrong adapter or suite")
        if item["model_id"] != run["model_id"] or item["seed"] != run["seed"]:
            raise ValueError(f"{label} evaluation has wrong model or seed")
        if item["adapter_sha256"] != run[f"{label}_adapter_hashes"]["adapter_model.safetensors"]:
            raise ValueError(f"{label} adapter hash mismatch")
        if item["heldout"]["attempts"] != expected_attempts:
            raise ValueError(f"{label} evaluation lacks {expected_attempts} attempts")
    if run["optimizer_steps"] < 1 or run["pre_adapter_hashes"]["adapter_model.safetensors"] == run["post_adapter_hashes"]["adapter_model.safetensors"]:
        raise ValueError("no verified LLM adapter update")

    error_patterns = ("memory allocation failed with oom", "out of memory", "traceback (most recent call last)")
    for label, path in (("pre", args.pre_log), ("post", args.post_log)):
        log = path.read_text(encoding="utf-8").lower()
        if any(pattern in log for pattern in error_patterns):
            raise ValueError(f"{label} evaluation log contains a memory or execution failure")
    before = _by_case(pre["heldout"]["rows"])
    after = _by_case(post["heldout"]["rows"])
    if before.keys() != after.keys() or before.keys() != expected_case_ids:
        raise ValueError("held-out case sets differ")
    if any(len(before[key]) != generations or len(after[key]) != generations for key in before):
        raise ValueError("each case must have the declared attempts per policy")
    per_case = []
    for case_id in sorted(before):
        per_case.append({
            "case_id": case_id,
            "before_success_rate": _mean(before[case_id], "success"),
            "after_success_rate": _mean(after[case_id], "success"),
            "before_labops": _mean(before[case_id], "labops"),
            "after_labops": _mean(after[case_id], "labops"),
            "before_researchsense": _mean(before[case_id], "researchsense"),
            "after_researchsense": _mean(after[case_id], "researchsense"),
            "before_reward": _mean(before[case_id], "reward"),
            "after_reward": _mean(after[case_id], "reward"),
        })
    deltas = [row["after_success_rate"] - row["before_success_rate"] for row in per_case]
    rng = random.Random(run["seed"])
    boot = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(2000))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": run["model_id"], "training_commit": run["code_commit"],
        "evaluation_code_commit": args.evaluation_code_commit,
        "suite_hash": run["suite_hash"], "scorer_version": run["scorer_version"],
        "seed": run["seed"], "optimizer_steps": run["optimizer_steps"],
        "evaluation_split": evaluation_split,
        "pre_adapter_sha256": pre["adapter_sha256"],
        "post_adapter_sha256": post["adapter_sha256"],
        "heldout_cases": len(per_case), "attempts_per_policy": expected_attempts,
        "before_successes": pre["heldout"]["successes"],
        "after_successes": post["heldout"]["successes"],
        "before_success_rate": pre["heldout"]["successes"] / expected_attempts,
        "after_success_rate": post["heldout"]["successes"] / expected_attempts,
        "observed_delta": sum(deltas) / len(deltas),
        "paired_case_bootstrap_95pct": [boot[49], boot[1949]],
        "regressed_cases": sum(delta < 0 for delta in deltas),
        "improved_cases": sum(delta > 0 for delta in deltas),
        "per_case": per_case,
        "interpretation": "Descriptive held-out comparison; no broad scientific reliability claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    comparison_json = output.with_suffix(".json")
    if comparison_json.exists():
        raise FileExistsError("comparison manifest already exists")
    comparison_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    source_files = [
        "backend/pyproject.toml", "backend/biogym/data/rl_snapshots_v1.json",
        "backend/biogym/rl_suite.py", "backend/biogym/rl_train.py",
        "backend/biogym/rl_eval.py", "backend/biogym/rl_compare.py",
        "notebooks/biogym_rl_colab.ipynb", "docs/rl-training.md",
    ]
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        _zip_tree(bundle, args.run_dir, "run")
        _zip_tree(bundle, args.pre_eval, "pre_eval")
        _zip_tree(bundle, args.post_eval, "post_eval")
        bundle.write(args.pre_log, "logs/pre_eval.log")
        bundle.write(args.post_log, "logs/post_eval.log")
        bundle.write(comparison_json, "comparison_manifest.json")
        for relative in source_files:
            path = args.repo_dir / relative
            bundle.write(path, f"code/{relative}")
    print(json.dumps({
        "before": report["before_successes"], "after": report["after_successes"],
        "cases": report["heldout_cases"], "attempts_per_policy": expected_attempts,
        "bundle": str(output), "bundle_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }))


if __name__ == "__main__":
    main()
