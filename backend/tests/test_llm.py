import json

import httpx
import pytest

from biogym.llm import SarvamActionProvider, run_trajectory
from biogym.curation import curate_successes
from biogym.models import Action, SourceStatus
from biogym.store import Store


class ScriptedProvider:
    model_id = "scripted-test-provider"

    def decide(self, observation, events):
        if observation["phase"] == "tool":
            return Action(action_id="script-tool", type="tool_call", name="validate_fasta"), {
                "fixture": True, "request_messages": [{"role": "system", "content": "fixture"}],
            }
        return Action(
            action_id="script-conclusion", type="conclude", claim="valid_sequence",
            evidence_ids=[events[-1]["event_id"]], source_status_claim=SourceStatus.FIXTURE,
        ), {"fixture": True, "request_messages": [{"role": "system", "content": "fixture"}]}


def test_trajectory_records_model_decisions_and_audited_events(tmp_path):
    store = Store(tmp_path / "trajectory.db")
    summary = run_trajectory(store, ScriptedProvider(), "seq-01", 8, tmp_path / "trajectories")
    payload = json.loads((tmp_path / "trajectories" / f"{summary['attempt_id']}.json").read_text())
    assert summary["success"] is True
    assert len(payload["decisions"]) == 2
    assert payload["events"][-1]["outcome"] == 1
    assert payload["model_id"] == "scripted-test-provider"
    manifest = curate_successes(tmp_path / "trajectories", tmp_path / "curated" / "train.jsonl")
    assert manifest["examples"] == 2
    assert manifest["trajectories"] == 1
    with pytest.raises(ValueError, match="restricted"):
        run_trajectory(store, ScriptedProvider(), "seq-03", 8, tmp_path / "trajectories")


def test_sarvam_adapter_parses_one_tool_action_without_exporting_key():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={
            "id": "response-1", "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call-1", "type": "function", "function": {
                    "name": "submit_action", "arguments": '{"type":"tool_call","name":"validate_fasta"}',
                }}],
            }}], "usage": {"prompt_tokens": 30, "completion_tokens": 8},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    action, record = SarvamActionProvider(api_key="test-secret", client=client).decide(
        {"phase": "tool", "prompt": "Validate FASTA"}, [],
    )
    assert action.name == "validate_fasta"
    assert record["response_id"] == "response-1"
    assert requests[0].headers["api-subscription-key"] == "test-secret"
    assert "test-secret" not in json.dumps(record)
