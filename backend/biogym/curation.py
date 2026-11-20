"""Filter verified training trajectories into an SFT warm-start corpus."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .export import verify_events


def curate_successes(trajectory_dir: Path, output_path: Path) -> dict:
    rows = []
    included_attempts = []
    excluded = {"non_train": 0, "failed": 0, "model_error": 0, "missing_messages": 0}
    for path in sorted(trajectory_dir.glob("*.json")):
        raw = path.read_bytes()
        trajectory = json.loads(raw)
        verify_events(trajectory["events"])
        if trajectory["split"] != "train":
            excluded["non_train"] += 1
            continue
        if not trajectory["success"] or trajectory["events"][-1]["outcome"] != 1:
            excluded["failed"] += 1
            continue
        if any("model_error" in step["model"] for step in trajectory["decisions"]):
            excluded["model_error"] += 1
            continue
        examples = []
        for step in trajectory["decisions"]:
            messages = step["model"].get("request_messages")
            if not messages:
                examples = []
                break
            action = {key: value for key, value in step["action"].items() if key != "action_id"}
            examples.append({
                "prompt": messages,
                "completion": json.dumps(action, sort_keys=True),
                "case_id": trajectory["case_id"],
                "attempt_id": trajectory["attempt_id"],
                "model_id": trajectory["model_id"],
                "prompt_version": trajectory["prompt_version"],
                "step_reward": step["reward"],
                "source_trajectory_sha256": hashlib.sha256(raw).hexdigest(),
            })
        if not examples:
            excluded["missing_messages"] += 1
            continue
        rows.extend(examples)
        included_attempts.append(trajectory["attempt_id"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    output_path.write_bytes(body)
    manifest = {
        "schema_version": "1.0.0", "kind": "verified-train-sft",
        "examples": len(rows), "trajectories": len(included_attempts),
        "attempt_ids": included_attempts, "excluded": excluded,
        "sha256": hashlib.sha256(body).hexdigest(),
        "limitations": "Successful authored training cases only; not held-out biological validation.",
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
