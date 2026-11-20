import json

from biogym.rl_suite import BioGymRLToolEnv, SOURCES, TASKS, split_tasks


def _tool_result(text: str) -> dict:
    return json.loads(text)


def test_source_splits_and_task_counts():
    assert len(SOURCES) == 24
    assert len(split_tasks("train")) == 64
    assert len(split_tasks("test")) == 16
    assert len(split_tasks("transfer")) == 16
    for left, right in (("train", "test"), ("train", "transfer"), ("test", "transfer")):
        assert {task.source_key for task in split_tasks(left)}.isdisjoint(
            {task.source_key for task in split_tasks(right)}
        )
    assert all("expected_claim" not in task.public() for task in TASKS.values())


def test_sequence_requires_valid_tools_evidence_and_provenance():
    env = BioGymRLToolEnv()
    task = TASKS["rl-seq-P69905-match"]
    observation = json.loads(env.reset(case_id=task.id, split="train"))
    assert "expected_claim" not in observation
    lookup = _tool_result(env.lookup_uniprot("P69905"))
    validation = _tool_result(env.check_fasta(task.input_data["fasta"]))
    result = _tool_result(env.submit_conclusion(
        "sequence_match", f"{lookup['evidence_id']},{validation['evidence_id']}",
        "CACHED", "database_metadata_only",
    ))
    assert result["verified_success"]
    assert env.done and env.success and env.reward > 1


def test_recovery_and_false_claim():
    env = BioGymRLToolEnv()
    task = TASKS["rl-seq-P69905-recovery"]
    env.reset(case_id=task.id, split="train")
    fault = _tool_result(env.lookup_uniprot("P69905"))
    assert fault["error"] == "transient_upstream_timeout"
    lookup = _tool_result(env.lookup_uniprot("P69905"))
    validation = _tool_result(env.check_fasta(task.input_data["fasta"]))
    result = _tool_result(env.submit_conclusion(
        "sequence_mismatch", f"{lookup['evidence_id']},{validation['evidence_id']}",
        "CACHED", "database_metadata_only",
    ))
    assert not result["verified_success"]
    assert result["evidence_complete"]
    assert env.done and not env.success


def test_structure_method_and_peptide_scope():
    env = BioGymRLToolEnv()
    env.reset(case_id="rl-struct-1VII-lookup", split="train")
    structure = _tool_result(env.lookup_rcsb("1VII"))
    assert structure["method"] == "SOLUTION NMR"
    result = _tool_result(env.submit_conclusion(
        "nmr", structure["evidence_id"], "CACHED", "database_metadata_only"
    ))
    assert result["verified_success"]

    task = TASKS["rl-epi-P69905-present"]
    env.reset(case_id=task.id, split="train")
    sequence = _tool_result(env.lookup_uniprot("P69905"))
    peptide = _tool_result(env.analyze_peptide(task.input_data["peptide"]))
    result = _tool_result(env.submit_conclusion(
        "computational_candidate", f"{sequence['evidence_id']},{peptide['evidence_id']}",
        "CACHED", "experimental_validation",
    ))
    assert not result["verified_success"]
    assert not result["provenance_correct"]
