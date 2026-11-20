import json
import hashlib

import pytest

from biogym.rl_demonstrations import build_demonstration, decision_examples
from biogym.rl_suite import SUITE_HASH, TASKS, split_tasks
from biogym.rl_train import validate_warm_adapter


def test_recovery_demonstration_contains_failed_call_retry_and_bounded_result():
    demo = build_demonstration(TASKS["rl-seq-P69905-recovery"])
    calls = [message["tool_calls"][0]["function"]["name"] for message in demo["messages"]
             if message["role"] == "assistant" and "tool_calls" in message]
    assert calls == ["lookup_uniprot", "lookup_uniprot", "check_fasta", "submit_conclusion"]
    results = [json.loads(message["content"]) for message in demo["messages"]
               if message["role"] == "tool"]
    assert results[0]["error"] == "transient_upstream_timeout"
    assert results[-1]["verified_success"] is True
    submitted = [message for message in demo["messages"]
                 if message["role"] == "assistant" and "tool_calls" in message][-1]
    args = submitted["tool_calls"][0]["function"]["arguments"]
    assert args["claim"] == "sequence_match"
    assert args["source_status"] == "CACHED"
    assert results[0]["evidence_id"] not in args["evidence_ids"]


def test_joint_demonstration_uses_three_tools_without_linking_records():
    task = next(task for task in split_tasks("train") if task.kind == "joint")
    demo = build_demonstration(task)
    calls = [message["tool_calls"][0]["function"]["name"] for message in demo["messages"]
             if message["role"] == "assistant" and "tool_calls" in message]
    assert calls == ["lookup_uniprot", "check_fasta", "lookup_rcsb", "submit_conclusion"]
    assert "independent" in demo["messages"][0]["content"].lower()
    result = json.loads([message["content"] for message in demo["messages"]
                         if message["role"] == "tool"][-1])
    assert result["verified_success"] is True


def test_decision_examples_keep_tool_results_in_prompt_and_agent_call_in_completion():
    demo = build_demonstration(TASKS["rl-seq-P69905-recovery"])
    examples = decision_examples(demo)
    assert len(examples) == 4
    assert examples[0]["prompt"] == demo["messages"][:1]
    assert examples[0]["completion"][0]["tool_calls"][0]["function"]["name"] == "lookup_uniprot"
    assert json.loads(examples[1]["prompt"][-1]["content"])["error"] == "transient_upstream_timeout"
    assert examples[-1]["completion"][0]["tool_calls"][0]["function"]["name"] == "submit_conclusion"
    assert examples[-1]["prompt"][-1]["role"] == "tool"


def test_demonstration_builder_rejects_heldout_cases():
    for split in ("test", "transfer"):
        with pytest.raises(ValueError, match="train"):
            build_demonstration(split_tasks(split)[0])


def test_grpo_warm_adapter_checks_model_suite_and_tensor_hash(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"trained tensor fixture")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    demonstrations = tmp_path / "train_demonstrations.jsonl"
    demonstrations.write_text('{"case_id":"rl-seq-P69905-match"}\n', encoding="utf-8")
    manifest = {
        "model_id": "test/model", "suite_hash": SUITE_HASH,
        "optimizer_steps": 2,
        "case_ids": ["rl-seq-P69905-match"],
        "adapter_hashes": {"adapter_model.safetensors": digest},
        "demonstration_sha256": hashlib.sha256(demonstrations.read_bytes()).hexdigest(),
    }
    (tmp_path / "warmstart_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_warm_adapter(adapter, "test/model")["optimizer_steps"] == 2
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        validate_warm_adapter(adapter, "test/model")
    with pytest.raises(ValueError, match="model"):
        validate_warm_adapter(adapter, "wrong/model")
    weights.write_bytes(b"trained tensor fixture")
    demonstrations.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="demonstration hash"):
        validate_warm_adapter(adapter, "test/model")
