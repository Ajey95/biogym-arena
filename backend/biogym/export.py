from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

from .cases import BY_ID
from .store import Store


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_events(events: list[dict]) -> None:
    previous = "GENESIS"
    for sequence, event in enumerate(events):
        if event["sequence"] != sequence or event["previous_hash"] != previous:
            raise ValueError("event chain is broken")
        payload = {key: value for key, value in event.items() if key != "hash"}
        if _sha(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()) != event["hash"]:
            raise ValueError("event hash mismatch")
        previous = event["hash"]


def _add_coordinate_artifacts(store: Store, events: list[dict], files: dict[str, bytes]) -> None:
    artifact_dir = Path(store.path).parent / "artifacts"
    for event in events:
        result = event.get("observation", {}).get("last_result") or {}
        artifact = result.get("artifact") or {}
        digest = artifact.get("sha256")
        if not digest:
            continue
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid coordinate artifact digest")
        path = artifact_dir / f"{digest}.pdb"
        raw = path.read_bytes()
        if _sha(raw) != digest or len(raw) != artifact.get("byte_length"):
            raise ValueError("coordinate artifact integrity mismatch")
        files[f"artifacts/{digest}.pdb"] = raw


def build_attempt_export(store: Store, attempt_id: str) -> bytes:
    row = store.attempt(attempt_id)
    if row is None:
        raise KeyError(attempt_id)
    events = store.events(attempt_id)
    verify_events(events)
    case = BY_ID[row["case_id"]]
    policy = store.policy(row["policy_id"])
    files = {
        "attempt.json": _json(row),
        "events.json": _json(events),
        "case.json": _json(case.model_dump(mode="json")),
        "README.txt": (
            b"BioGym Arena recorded-run archive.\n"
            b"Replay events.json in sequence and verify each event hash and previous_hash.\n"
            b"FIXTURE inputs are authored exercises; they are not experimental validation.\n"
        ),
    }
    _add_coordinate_artifacts(store, events, files)
    if policy:
        files["policy.json"] = _json(policy)
        path = Path(policy["artifact_path"])
        if path.exists() and _sha(path.read_bytes()) == policy["sha256"]:
            files[f"policies/{path.name}"] = path.read_bytes()
    return _make_zip(files, {"kind": "attempt", "attempt_id": attempt_id,
                             "offline_replay_verified": True,
                             "computational_rerun": "conditional"})


def build_evaluation_export(store: Store, evaluation_id: str) -> bytes:
    report = store.evaluation(evaluation_id)
    if report is None:
        raise KeyError(evaluation_id)
    files = {"evaluation.json": _json(report)}
    for row in report["rows"]:
        attempt_id = row["attempt_id"]
        attempt = store.attempt(attempt_id)
        events = store.events(attempt_id)
        if attempt is None:
            raise ValueError(f"missing attempt {attempt_id}")
        verify_events(events)
        files[f"attempts/{attempt_id}/attempt.json"] = _json(attempt)
        files[f"attempts/{attempt_id}/events.json"] = _json(events)
        files[f"cases/{attempt['case_id']}.json"] = _json(BY_ID[attempt["case_id"]].model_dump(mode="json"))
        _add_coordinate_artifacts(store, events, files)
    for policy_id in (report["baseline_policy_id"], report["trained_policy_id"]):
        policy = store.policy(policy_id)
        if policy is None:
            raise ValueError(f"missing policy {policy_id}")
        files[f"policies/{policy_id}.json"] = _json(policy)
        path = Path(policy["artifact_path"])
        if path.exists() and _sha(path.read_bytes()) == policy["sha256"]:
            files[f"policies/{path.name}"] = path.read_bytes()
    files["README.txt"] = (
        b"BioGym Arena evaluation archive.\n"
        b"All recorded attempts and event chains are included for offline inspection.\n"
        b"Policy checkpoints are included when available. This starter policy requires the recorded code and PyTorch for a computational rerun.\n"
        b"All initial scientific cases are FIXTURE exercises, not experimental validation.\n"
    )
    return _make_zip(files, {"kind": "evaluation", "evaluation_id": evaluation_id,
                             "offline_replay_verified": True,
                             "computational_rerun": "conditional"})


def _make_zip(files: dict[str, bytes], metadata: dict) -> bytes:
    manifest = {
        "schema_version": "1.0.0", **metadata,
        "files": {name: _sha(content) for name, content in sorted(files.items())},
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("manifest.json", _json(manifest))
    return buffer.getvalue()


def verify_zip(data: bytes) -> dict:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for name, expected in manifest["files"].items():
            if _sha(archive.read(name)) != expected:
                raise ValueError(f"artifact hash mismatch: {name}")
        for name in manifest["files"]:
            if name.endswith("/events.json") or name == "events.json":
                verify_events(json.loads(archive.read(name)))
    return manifest
