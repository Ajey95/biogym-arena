import json

from biogym.rl_suite import BioGymRLToolEnv, SOURCES, TASKS, split_tasks


def _tool_result(text: str) -> dict:
    return json.loads(text)


def test_source_splits_and_task_counts():
    assert len(SOURCES) == 24
    assert len(split_tasks("train")) == 80
    assert len(split_tasks("test")) == 20
    assert len(split_tasks("transfer")) == 20
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
        "sequence_window_contains_peptide", f"{sequence['evidence_id']},{peptide['evidence_id']}",
        "CACHED", "experimental_validation",
    ))
    assert not result["verified_success"]
    assert not result["provenance_correct"]


def test_observation_does_not_supply_scoring_answer_or_limitation():
    task = TASKS["rl-epi-P69905-present"]
    observation = json.loads(BioGymRLToolEnv().reset(case_id=task.id))
    assert "expected_claim" not in observation
    assert "required_limitation" not in observation
    assert "predictor_score" not in observation["input"]
    assert "claim_options" not in observation


def test_repeated_successful_lookup_cannot_increase_process_reward():
    env = BioGymRLToolEnv()
    env.reset(case_id="rl-struct-1VII-lookup")
    env.lookup_rcsb("1VII")
    first_reward = env.reward
    env.lookup_rcsb("1VII")
    assert env.reward <= first_reward


def test_peptide_check_uses_cached_sequence_not_authored_score():
    env = BioGymRLToolEnv()
    task = TASKS["rl-epi-P69905-present"]
    env.reset(case_id=task.id)
    result = _tool_result(env.analyze_peptide(task.input_data["peptide"]))
    assert result["in_sequence_window"] is True
    assert result["experimental_validation"] is False
    assert result["source_status"] == "CACHED"
    assert "score" not in result

    task = TASKS["rl-epi-P69905-absent"]
    env.reset(case_id=task.id)
    result = _tool_result(env.analyze_peptide(task.input_data["peptide"]))
    assert result["in_sequence_window"] is False


def test_peptide_task_claims_only_window_occurrence():
    assert TASKS["rl-epi-P69905-present"].expected_claim == "sequence_window_contains_peptide"
    assert TASKS["rl-epi-P69905-absent"].expected_claim == "sequence_window_lacks_peptide"
    assert TASKS["rl-epi-P69905-present"].limitation == "sequence_occurrence_not_validation"


def test_unsupported_binding_claim_requires_explicit_abstention():
    env = BioGymRLToolEnv()
    task = TASKS["rl-epi-P69905-binding_claim"]
    env.reset(case_id=task.id)
    lookup = _tool_result(env.lookup_uniprot("P69905"))
    peptide = _tool_result(env.analyze_peptide(task.input_data["peptide"]))
    cited = f"{lookup['evidence_id']},{peptide['evidence_id']}"
    result = _tool_result(env.submit_conclusion(
        "sequence_window_contains_peptide", cited, "CACHED", "sequence_occurrence_not_validation"
    ))
    assert not result["verified_success"]

    env.reset(case_id=task.id)
    lookup = _tool_result(env.lookup_uniprot("P69905"))
    peptide = _tool_result(env.analyze_peptide(task.input_data["peptide"]))
    result = _tool_result(env.submit_conclusion(
        "insufficient_experimental_evidence",
        f"{lookup['evidence_id']},{peptide['evidence_id']}",
        "CACHED", "sequence_occurrence_not_validation",
    ))
    assert result["verified_success"]


def test_transfer_joint_audit_requires_evidence_from_both_rooms():
    task = next(task for task in split_tasks("transfer") if task.kind == "joint")
    observation = json.loads(BioGymRLToolEnv().reset(case_id=task.id, split="transfer"))
    assert observation["room"] == "transfer"
    assert "accession" in observation["input"] and "pdb_id" in observation["input"]
    env = BioGymRLToolEnv()
    env.reset(case_id=task.id, split="transfer")
    sequence = _tool_result(env.lookup_uniprot(task.input_data["accession"]))
    fasta = _tool_result(env.check_fasta(task.input_data["fasta"]))
    missing_structure = _tool_result(env.submit_conclusion(
        task.expected_claim, f"{sequence['evidence_id']},{fasta['evidence_id']}",
        "CACHED", "independent_metadata_only",
    ))
    assert not missing_structure["verified_success"]

    env.reset(case_id=task.id, split="transfer")
    sequence = _tool_result(env.lookup_uniprot(task.input_data["accession"]))
    fasta = _tool_result(env.check_fasta(task.input_data["fasta"]))
    structure = _tool_result(env.lookup_rcsb(task.input_data["pdb_id"]))
    result = _tool_result(env.submit_conclusion(
        task.expected_claim,
        ",".join(x["evidence_id"] for x in (sequence, fasta, structure)),
        "CACHED", "independent_metadata_only",
    ))
    assert result["verified_success"]


def test_every_generated_case_is_reachable_within_its_action_budget():
    for task in TASKS.values():
        env = BioGymRLToolEnv()
        env.reset(case_id=task.id, split=task.split)
        evidence_ids = []
        for tool in task.required_tools:
            argument = {
                "lookup_uniprot": task.input_data.get("accession"),
                "check_fasta": task.input_data.get("fasta"),
                "lookup_rcsb": task.input_data.get("pdb_id"),
                "analyze_peptide": task.input_data.get("peptide"),
            }[tool]
            result = _tool_result(getattr(env, tool)(argument))
            if result.get("error") == "transient_upstream_timeout":
                result = _tool_result(getattr(env, tool)(argument))
            evidence_ids.append(result["evidence_id"])
        finding = _tool_result(env.submit_conclusion(
            task.expected_claim, ",".join(evidence_ids), "CACHED", task.limitation,
        ))
        assert finding["verified_success"], task.id
        assert env.done and len(env._events) <= task.max_actions, task.id
