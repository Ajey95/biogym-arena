"""Verified train-split tool trajectories for an LLM tool-use warm start.

These demonstrations come from the versioned BioGym verifier, not from an
experimental assay. Held-out task IDs and expected answers are rejected.
"""
from __future__ import annotations

import json

from .rl_suite import BioGymRLToolEnv, SUITE_HASH, Task


def _tool_arguments(task: Task, name: str) -> dict[str, str]:
    field = {
        "lookup_uniprot": ("accession", "accession"),
        "check_fasta": ("fasta", "fasta"),
        "lookup_rcsb": ("pdb_id", "pdb_id"),
        "analyze_peptide": ("peptide", "peptide"),
    }[name]
    return {field[0]: task.input_data[field[1]]}


def _call_message(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {"name": name, "arguments": arguments}}
    ]}


def build_demonstration(task: Task) -> dict:
    """Execute one successful, source-labeled demonstration from the train split."""
    if task.split != "train":
        raise ValueError("demonstrations may use train cases only")
    env = BioGymRLToolEnv()
    observation = env.reset(case_id=task.id, split="train")
    messages = [{
        "role": "user",
        "content": "Complete the BioGym scientific task in the environment. "
                   "Use tool evidence for your final decision.\n" + observation,
    }]
    evidence_ids = []
    for name in task.required_tools:
        arguments = _tool_arguments(task, name)
        while True:
            messages.append(_call_message(name, arguments))
            result_text = getattr(env, name)(**arguments)
            messages.append({"role": "tool", "name": name, "content": result_text})
            result = json.loads(result_text)
            if result.get("error") == "transient_upstream_timeout":
                continue
            if not result.get("ok"):
                raise RuntimeError(f"demonstration tool failed: {task.id} {name}")
            evidence_ids.append(result["evidence_id"])
            break
    conclusion = {
        "claim": task.expected_claim,
        "evidence_ids": ",".join(evidence_ids),
        "source_status": "CACHED",
        "limitation": task.limitation,
    }
    messages.append(_call_message("submit_conclusion", conclusion))
    result_text = env.submit_conclusion(**conclusion)
    messages.append({"role": "tool", "name": "submit_conclusion", "content": result_text})
    if not env.success:
        raise RuntimeError(f"demonstration did not verify: {task.id}")
    messages.append({"role": "assistant", "content": "Submitted a bounded conclusion with cited evidence."})
    return {"suite_hash": SUITE_HASH, "case_id": task.id,
            "source_key": task.source_key, "messages": messages,
            "verified_success": True}
