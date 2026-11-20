import hashlib
import json
import sys
from pathlib import Path

import pytest

from biogym.rl_compare import main
from biogym.rl_suite import SUITE_HASH, TASKS, split_tasks


def _prepare_comparison(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    pre_dir = tmp_path / "pre"
    post_dir = tmp_path / "post"
    for directory in (run_dir, pre_dir, post_dir):
        directory.mkdir()
    cases = [task.id for task in split_tasks("transfer")]
    assert len(cases) == 20
    run = {
        "suite_hash": SUITE_HASH, "model_id": "test/model", "seed": 7,
        "code_commit": "test-commit", "scorer_version": "source-evidence-rubric-v2",
        "optimizer_steps": 1,
        "pre_adapter_hashes": {"adapter_model.safetensors": "before"},
        "post_adapter_hashes": {"adapter_model.safetensors": "after"},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run), encoding="utf-8")
    for label, directory, adapter in (("pre", pre_dir, "before"), ("post", post_dir, "after")):
        rows = []
        traces = directory / "traces"
        traces.mkdir()
        for case_index, case_id in enumerate(cases):
            for attempt_index in range(2):
                attempt_id = hashlib.sha256(
                    f"{label}-{case_index}-{attempt_index}".encode()
                ).hexdigest()[:32]
                row = {"case_id": case_id, "attempt_id": attempt_id,
                       "success": label == "post", "done": True,
                       "labops": 1.0, "researchsense": 1.0, "reward": 1.0}
                rows.append(row)
                trace = {**row, "schema_version": "biogym-rl-v2", "suite_hash": SUITE_HASH,
                         "split": "transfer", "source_key": TASKS[case_id].source_key,
                         "events": []}
                (traces / f"{attempt_id}.json").write_text(json.dumps(trace), encoding="utf-8")
        item = {
            "adapter": label, "suite_hash": SUITE_HASH, "model_id": "test/model",
            "seed": 7, "adapter_sha256": adapter, "split": "transfer",
            "heldout": {"attempts": 40, "successes": 40 if label == "post" else 0,
                        "rows": rows},
        }
        (directory / "evaluation_manifest.json").write_text(json.dumps(item), encoding="utf-8")
    pre_log = tmp_path / "pre.log"
    post_log = tmp_path / "post.log"
    pre_log.write_text("evaluation complete", encoding="utf-8")
    post_log.write_text("evaluation complete", encoding="utf-8")
    bundle = tmp_path / "comparison.zip"
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(sys, "argv", [
        "rl_compare", "--run-dir", str(run_dir), "--pre-eval", str(pre_dir),
        "--post-eval", str(post_dir), "--pre-log", str(pre_log),
        "--post-log", str(post_log), "--repo-dir", str(repo),
        "--evaluation-code-commit", "eval-commit", "--output", str(bundle),
    ])
    return bundle, pre_dir


def test_transfer_comparison_accepts_complete_versioned_case_set(tmp_path, monkeypatch):
    bundle, _ = _prepare_comparison(tmp_path, monkeypatch)
    main()
    report = json.loads(bundle.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["evaluation_split"] == "transfer"
    assert report["heldout_cases"] == 20
    assert report["attempts_per_policy"] == 40
    assert bundle.is_file()


def test_comparison_rejects_trace_that_disagrees_with_report(tmp_path, monkeypatch):
    _, pre_dir = _prepare_comparison(tmp_path, monkeypatch)
    attempt_id = hashlib.sha256(b"pre-0-0").hexdigest()[:32]
    trace_path = pre_dir / "traces" / f"{attempt_id}.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    trace["reward"] = 999.0
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    with pytest.raises(ValueError, match="trace"):
        main()
