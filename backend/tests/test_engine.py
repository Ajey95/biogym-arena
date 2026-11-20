from biogym.engine import EpisodeEngine
from biogym.models import Action, SourceStatus
from biogym.store import Store


def engine(tmp_path):
    return EpisodeEngine(Store(tmp_path / "test.db"))


def test_success_requires_evidence_and_correct_source(tmp_path):
    runtime = engine(tmp_path)
    attempt = runtime.reset("seq-01")
    tool = runtime.step(attempt.id, Action(action_id="tool", type="tool_call", name="validate_fasta"))
    assert tool.observation["last_result"]["result"]["valid"] is True
    final = runtime.step(attempt.id, Action(
        action_id="final", type="conclude", claim="valid_sequence",
        evidence_ids=[tool.event_id], source_status_claim=SourceStatus.FIXTURE,
    ))
    assert final.done and final.components["outcome"] == 1
    assert final.components["researchsense"] == 1


def test_wrong_provenance_fails_even_with_correct_claim(tmp_path):
    runtime = engine(tmp_path)
    attempt = runtime.reset("epi-01")
    tool = runtime.step(attempt.id, Action(action_id="tool", type="tool_call", name="analyze_epitope"))
    final = runtime.step(attempt.id, Action(
        action_id="final", type="conclude", claim="computational_candidate",
        evidence_ids=[tool.event_id], source_status_claim=SourceStatus.LIVE,
    ))
    assert final.components["outcome"] == 0
    assert final.observation["finding"]["provenance_correct"] is False


def test_duplicate_action_is_idempotent(tmp_path):
    runtime = engine(tmp_path)
    attempt = runtime.reset("seq-01")
    action = Action(action_id="same", type="tool_call", name="validate_fasta")
    first = runtime.step(attempt.id, action)
    second = runtime.step(attempt.id, action)
    assert first.event_id == second.event_id
    assert runtime.state(attempt.id).step_count == 1


def test_fault_can_be_retried_and_failure_is_recorded(tmp_path):
    runtime = engine(tmp_path)
    attempt = runtime.reset("seq-02")
    first = runtime.step(attempt.id, Action(action_id="a", type="tool_call", name="validate_fasta"))
    assert first.observation["last_result"]["error"] == "injected_transient_failure"
    second = runtime.step(attempt.id, Action(action_id="b", type="tool_call", name="validate_fasta"))
    assert second.observation["last_result"]["result"]["valid"] is False
    final = runtime.step(attempt.id, Action(
        action_id="c", type="conclude", claim="invalid_sequence",
        evidence_ids=[second.event_id], source_status_claim=SourceStatus.FIXTURE,
    ))
    assert final.components["outcome"] == 1
    assert len(runtime.state(attempt.id).events) == 4


def test_budget_exhaustion_and_invalid_call(tmp_path):
    runtime = engine(tmp_path)
    attempt = runtime.reset("seq-01")
    for index in range(5):
        result = runtime.step(attempt.id, Action(
            action_id=str(index), type="tool_call", name="not_a_tool",
        ))
    assert result.done and result.terminal_reason == "budget_exhausted"
    assert result.observation["last_result"]["error"] == "unknown_tool"
