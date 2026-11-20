from __future__ import annotations

import uuid

from .cases import BY_ID, CASES
from .engine import EpisodeEngine
from .policy import load_policy, rollout
from .store import Store


def evaluate(
    store: Store, engine: EpisodeEngine, baseline_id: str, trained_id: str,
    seed: int = 1000, repeats: int = 3,
) -> dict:
    if repeats < 1 or repeats > 30:
        raise ValueError("repeats must be between 1 and 30")
    baseline = store.policy(baseline_id)
    trained = store.policy(trained_id)
    if baseline is None or trained is None:
        raise KeyError("policy not found")
    if trained["parent_id"] != baseline_id or trained["optimizer_steps"] < 1:
        raise ValueError("comparison has no recorded policy update")
    cases = [case for case in CASES if case.split in ("test", "transfer")]
    rows = []
    for case in cases:
        for repeat in range(repeats):
            run_seed = seed + repeat
            for policy in (baseline, trained):
                reward, _, attempt_id, success = rollout(
                    engine, load_policy(policy), case.id, run_seed, policy["id"], sample=False,
                )
                state = engine.state(attempt_id)
                final = state.events[-1]
                rows.append({
                    "case_id": case.id, "split": case.split, "room": case.room,
                    "source_status": case.source_status.value, "case_version": case.version,
                    "seed": run_seed, "policy_id": policy["id"],
                    "attempt_id": attempt_id, "success": success, "reward": reward,
                    "labops": final.labops, "researchsense": final.researchsense,
                    "steps": state.step_count, "terminal_reason": state.observation.get("terminal_reason"),
                })
    summary = {}
    for policy in (baseline, trained):
        selected = [row for row in rows if row["policy_id"] == policy["id"]]
        summary[policy["id"]] = {
            "successes": sum(row["success"] for row in selected), "attempts": len(selected),
            "success_rate": sum(row["success"] for row in selected) / len(selected),
            "mean_reward": sum(row["reward"] for row in selected) / len(selected),
            "mean_labops": sum(row["labops"] for row in selected) / len(selected),
            "mean_researchsense": sum(row["researchsense"] for row in selected) / len(selected),
        }
    evaluation_id = f"eval-{uuid.uuid4().hex[:10]}"
    report = {
        "id": evaluation_id, "label": "baseline vs after policy-gradient training",
        "baseline_policy_id": baseline_id, "trained_policy_id": trained_id,
        "baseline_hash": baseline["sha256"], "trained_hash": trained["sha256"],
        "scorer_version": "episode-rubric-v1", "seed": seed, "repeats": repeats,
        "case_ids": [case.id for case in cases], "rows": rows, "summary": summary,
        "limitations": [
            "Starter policy is a small neural action policy, not a language model.",
            "Initial cases are authored FIXTURE exercises, not biological validation.",
            "Repeated deterministic evaluations do not create independent biological cases.",
        ],
    }
    store.put_evaluation(evaluation_id, report)
    return report
