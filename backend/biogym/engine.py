from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .cases import BY_ID
from .models import Action, AttemptState, Event, StepResult
from .store import Store
from .tools import TOOL_NAMES, run_scientific_tool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_event(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class EpisodeEngine:
    def __init__(self, store: Store):
        self.store = store

    def reset(self, case_id: str, policy_id: str = "manual", seed: int = 0) -> AttemptState:
        case = BY_ID.get(case_id)
        if case is None:
            raise KeyError(case_id)
        attempt_id = uuid.uuid4().hex
        observation = {
            "phase": "tool", "room": case.room, "mode": case.mode, "prompt": case.prompt,
            "input_data": case.input_data, "allowed_tools": list(TOOL_NAMES),
            "source_status": case.source_status.value, "last_result": None,
            "remaining_steps": case.max_steps,
        }
        self.store.insert_attempt({
            "id": attempt_id, "case_id": case_id, "policy_id": policy_id, "seed": seed,
            "status": "RUNNING", "step_count": 0, "observation": observation,
            "created_at": _now(),
        })
        event = self._event(attempt_id, 0, "reset", None, observation, 0, 0, 0, 0, "GENESIS")
        self.store.append(attempt_id, event, "RUNNING", observation)
        return self.state(attempt_id)

    def state(self, attempt_id: str) -> AttemptState:
        row = self.store.attempt(attempt_id)
        if row is None:
            raise KeyError(attempt_id)
        return AttemptState(
            id=row["id"], case_id=row["case_id"], policy_id=row["policy_id"], seed=row["seed"],
            status=row["status"], step_count=row["step_count"],
            max_steps=BY_ID[row["case_id"]].max_steps,
            observation=row["observation"],
            events=[Event.model_validate(e) for e in self.store.events(attempt_id)],
        )

    def step(self, attempt_id: str, action: Action) -> StepResult:
        old = self.store.action_event(attempt_id, action.action_id)
        if old is not None:
            state = self.state(attempt_id)
            return self._result(old, state)
        row = self.store.attempt(attempt_id)
        if row is None:
            raise KeyError(attempt_id)
        if row["status"] != "RUNNING":
            raise ValueError("attempt is terminal")
        case = BY_ID[row["case_id"]]
        events = self.store.events(attempt_id)
        sequence = row["step_count"] + 1
        prior_success = next((
            e for e in reversed(events)
            if e["kind"] == "tool_result"
            and e["observation"].get("last_result", {}).get("ok")
            and e["observation"]["last_result"].get("tool") == case.required_tool
        ), None)
        observation = dict(row["observation"])
        kind = "invalid_action"
        outcome = labops = researchsense = 0.0
        reward = -0.2
        terminal_reason = None

        if action.type == "tool_call":
            if action.name not in TOOL_NAMES:
                observation["last_result"] = {"ok": False, "error": "unknown_tool"}
            elif action.name != case.required_tool:
                kind = "tool_result"
                observation["last_result"] = {
                    "ok": False, "error": "wrong_tool_for_task", "tool": action.name,
                    "source_status": case.source_status.value,
                }
            else:
                earlier_calls = sum(
                    1 for e in events if (e.get("action") or {}).get("name") == action.name
                )
                if case.fault_once and earlier_calls == 0:
                    kind = "tool_result"
                    observation["last_result"] = {
                        "ok": False, "error": "injected_transient_failure",
                        "tool": action.name, "source_status": case.source_status.value,
                    }
                    reward = -0.05
                else:
                    kind = "tool_result"
                    try:
                        observation["last_result"] = run_scientific_tool(
                            case, action.name, Path(self.store.path).parent / "artifacts",
                        )
                    except Exception as error:
                        observation["last_result"] = {
                            "ok": False, "error": type(error).__name__,
                            "detail": str(error), "tool": action.name,
                            "source_status": "UNKNOWN",
                        }
                    if observation["last_result"].get("ok"):
                        observation["phase"] = "conclude"
                        labops = 1.0
                        reward = 0.2
                    else:
                        reward = -0.05
        elif action.type == "conclude":
            kind = "conclusion"
            evidence_ok = prior_success is not None and prior_success["event_id"] in action.evidence_ids
            recorded_status = (prior_success["observation"]["last_result"] or {}).get("source_status") if prior_success else None
            source_ok = action.source_status_claim is not None and action.source_status_claim.value == recorded_status
            claim_ok = action.claim == case.expected_claim
            labops = 1.0 if prior_success is not None else 0.0
            researchsense = 1.0 if evidence_ok and source_ok and claim_ok else 0.0
            outcome = 1.0 if evidence_ok and source_ok and claim_ok else 0.0
            reward = outcome + 0.2 * labops + 0.3 * researchsense - 0.02 * sequence
            terminal_reason = "verified_success" if outcome else "failed_verification"
            observation["finding"] = {
                "claim_supported": claim_ok and evidence_ok,
                "provenance_correct": source_ok,
                "evidence_linked": evidence_ok,
            }
        elif action.type == "abstain":
            kind = "abstention"
            terminal_reason = "abstained"
            reward = 0.0
            observation["finding"] = {"abstained": True, "reason": "case requires a verifiable conclusion"}

        observation["remaining_steps"] = case.max_steps - sequence
        if terminal_reason is None and sequence >= case.max_steps:
            terminal_reason = "budget_exhausted"
        if terminal_reason is not None:
            observation["phase"] = "terminal"
            observation["terminal_reason"] = terminal_reason
        status = "RUNNING" if terminal_reason is None else (
            "TERMINAL_SUCCESS" if outcome else "TERMINAL_FAILURE"
        )
        event = self._event(
            attempt_id, sequence, kind, action, observation,
            reward, labops, researchsense, outcome, events[-1]["hash"],
        )
        self.store.append(attempt_id, event, status, observation)
        return self._result(event, self.state(attempt_id))

    def _event(
        self, attempt_id: str, sequence: int, kind: str, action: Action | None,
        observation: dict, reward: float, labops: float, researchsense: float,
        outcome: float, previous_hash: str,
    ) -> dict:
        payload = {
            "event_id": uuid.uuid4().hex, "attempt_id": attempt_id, "sequence": sequence,
            "kind": kind, "action_id": action.action_id if action else None,
            "action": action.model_dump(mode="json") if action else None,
            "observation": observation, "reward": reward, "labops": labops,
            "researchsense": researchsense, "outcome": outcome,
            "previous_hash": previous_hash, "created_at": _now(),
        }
        payload["hash"] = _hash_event(payload)
        return payload

    def _result(self, event: dict, state: AttemptState) -> StepResult:
        return StepResult(
            attempt_id=state.id, event_id=event["event_id"],
            observation=event["observation"], reward=event["reward"],
            components={
                "outcome": event["outcome"], "labops": event["labops"],
                "researchsense": event["researchsense"],
            },
            done=state.status != "RUNNING" or event["kind"] in ("conclusion", "abstention"),
            terminal_reason=state.observation.get("terminal_reason"),
            remaining_steps=state.max_steps - state.step_count,
        )
