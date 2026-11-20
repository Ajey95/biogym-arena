"""Versioned, source-backed BioGym episodes for multi-turn LLM reinforcement learning.

The policy sees only reset observations and tool outputs. Expected answers stay in
this verifier module; model prompts and exported train rows never contain them.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path


SCHEMA_VERSION = "biogym-rl-v2"
SCORER_VERSION = "source-evidence-rubric-v2"
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_sources() -> dict[str, dict]:
    path = files("biogym").joinpath("data/rl_snapshots_v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["schema_version"] != "1.0.0":
        raise ValueError("unsupported source snapshot version")
    records = payload["sources"]
    keys = [(record["kind"], record["id"]) for record in records]
    if len(keys) != len(set(keys)) or any(len(record["raw_sha256"]) != 64 for record in records):
        raise ValueError("invalid source snapshot manifest")
    return {f"{record['kind']}:{record['id']}": record for record in records}


@dataclass(frozen=True)
class Task:
    id: str
    split: str
    room: str
    mode: str
    kind: str
    source_key: str
    input_data: dict
    expected_claim: str
    required_tools: tuple[str, ...]
    limitation: str
    prompt: str = ""
    secondary_source_key: str | None = None
    fault_once: bool = False
    max_actions: int = 6

    def public(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION, "room": self.room, "mode": self.mode,
            "input": self.input_data, "allowed_tools": list(self.required_tools),
            "task": self.prompt,
            "instruction": "Use tool evidence for the decision. Cite successful evidence IDs, "
                           "report the source status returned by tools, and give a scientific scope limitation. "
                           "Retry a transient database error within budget.",
            "remaining_actions": self.max_actions,
        }


def _mutate(sequence: str) -> str:
    replacement = "A" if sequence[7] != "A" else "C"
    return sequence[:7] + replacement + sequence[8:]


def make_tasks(sources: dict[str, dict] | None = None) -> dict[str, Task]:
    sources = sources or load_sources()
    tasks: dict[str, Task] = {}
    for source_key, source in sources.items():
        accession = source["id"]
        split = source["split"]
        if source["kind"] == "uniprot":
            segment = source["segment"]
            for variant, sequence, claim, fault in (
                ("match", segment, "sequence_match", False),
                ("mismatch", _mutate(segment), "sequence_mismatch", False),
                ("invalid", segment[:5] + "X" + segment[6:], "invalid_fasta", False),
                ("recovery", segment, "sequence_match", True),
            ):
                case_id = f"rl-seq-{accession}-{variant}"
                tasks[case_id] = Task(
                    case_id, split, "sequence", "ChaosGym" if fault else "PipelineGym",
                    "sequence", source_key,
                    {"accession": accession, "fasta": f">{accession}\n{sequence}"},
                    claim, ("lookup_uniprot", "check_fasta"), "database_metadata_only",
                    prompt="Validate the supplied FASTA, then compare its sequence with the "
                           "cached UniProt sequence window. Report match, mismatch, or invalid FASTA.",
                    fault_once=fault,
                )
            for variant, peptide, claim in (
                ("present", segment[3:12], "sequence_window_contains_peptide"),
                ("absent", _mutate(segment)[3:12], "sequence_window_lacks_peptide"),
            ):
                if (peptide in segment) != (variant == "present"):
                    raise ValueError(f"peptide case cannot be verified from cached window: {accession}")
                case_id = f"rl-epi-{accession}-{variant}"
                tasks[case_id] = Task(
                    case_id, split, "epitope", "EvidenceGym", "peptide", source_key,
                    {"accession": accession, "peptide": peptide},
                    claim, ("lookup_uniprot", "analyze_peptide"),
                    "sequence_occurrence_not_validation",
                    prompt="Check whether the peptide occurs in the cached UniProt sequence window. "
                           "Sequence occurrence is not binding, immunogenicity, or experimental validation.",
                )
            case_id = f"rl-epi-{accession}-binding_claim"
            tasks[case_id] = Task(
                case_id, split, "epitope", "EvidenceGym", "peptide", source_key,
                {"accession": accession, "peptide": segment[3:12],
                 "proposed_claim": "This peptide is experimentally validated to bind."},
                "insufficient_experimental_evidence", ("lookup_uniprot", "analyze_peptide"),
                "sequence_occurrence_not_validation",
                prompt="Audit the proposed binding claim using the available evidence. "
                       "Abstain if sequence metadata cannot establish experimental binding.",
            )
        else:
            method = source["method"].upper()
            claim = ("xray" if "X-RAY" in method else
                     "nmr" if "NMR" in method else
                     "electron_microscopy" if "ELECTRON MICROSCOPY" in method else None)
            if claim is None:
                raise ValueError(f"unsupported method for {accession}: {method}")
            for variant, fault in (("lookup", False), ("recovery", True)):
                case_id = f"rl-struct-{accession}-{variant}"
                tasks[case_id] = Task(
                    case_id, split, "structure", "ChaosGym" if fault else "ToolGym",
                    "structure", source_key, {"pdb_id": accession}, claim,
                    ("lookup_rcsb",), "database_metadata_only", fault_once=fault,
                    prompt="Identify the experimental method recorded for this RCSB entry. "
                           "Do not infer an unrelated biological result from its method metadata.",
                    max_actions=5,
                )
    for split in ("train", "test", "transfer"):
        proteins = [key for key, source in sources.items()
                    if source["split"] == split and source["kind"] == "uniprot"]
        structures = [key for key, source in sources.items()
                      if source["split"] == split and source["kind"] == "rcsb"]
        for protein_key, structure_key in zip(proteins, structures):
            protein = sources[protein_key]
            structure = sources[structure_key]
            method = structure["method"].upper()
            method_claim = ("xray" if "X-RAY" in method else "nmr" if "NMR" in method
                            else "electron_microscopy" if "ELECTRON MICROSCOPY" in method else None)
            if method_claim is None:
                raise ValueError(f"unsupported method for {structure['id']}: {method}")
            case_id = f"rl-joint-{protein['id']}-{structure['id']}"
            tasks[case_id] = Task(
                case_id, split, "transfer" if split == "transfer" else "cross_room",
                "ResearchGym", "joint", protein_key,
                {"accession": protein["id"], "pdb_id": structure["id"],
                 "fasta": f">{protein['id']}\n{protein['segment']}"},
                f"sequence_match|{method_claim}",
                ("lookup_uniprot", "check_fasta", "lookup_rcsb"),
                "independent_metadata_only",
                prompt="Audit two independent records. Compare the FASTA to its UniProt window "
                       "and identify the RCSB experimental method. Join the sequence and method "
                       "labels with '|'. Do not infer that the two records are biologically linked.",
                secondary_source_key=structure_key,
                max_actions=6,
            )
    if len(tasks) != len(set(tasks)):
        raise ValueError("duplicate task IDs")
    return tasks


SOURCES = load_sources()
TASKS = make_tasks(SOURCES)
SUITE_HASH = _digest({"sources": SOURCES, "tasks": [vars(t) for t in TASKS.values()]})


def split_tasks(split: str) -> list[Task]:
    if split not in {"train", "test", "transfer"}:
        raise ValueError("unknown split")
    return [task for task in TASKS.values() if task.split == split]


class BioGymRLToolEnv:
    """TRL environment_factory adapter with audited, stateful BioGym episodes."""

    def __init__(self):
        self._task: Task | None = None
        self._events: list[dict] = []
        self._evidence: dict[str, dict] = {}
        self._attempt_id = ""
        self._fault_fired = False
        self.reward = 0.0
        self.done = False
        self.success = False
        self.labops = 0.0
        self.researchsense = 0.0

    def reset(self, case_id: str, split: str = "train", **kwargs) -> str:
        task = TASKS[case_id]
        if task.split != split:
            raise ValueError("case is not assigned to requested split")
        self._task = task
        self._events = []
        self._evidence = {}
        self._attempt_id = uuid.uuid4().hex
        self._fault_fired = False
        self.reward = 0.0
        self.done = False
        self.success = False
        self.labops = 0.0
        self.researchsense = 0.0
        self._save_trace()
        return json.dumps(task.public(), sort_keys=True)

    def _require(self, name: str, supplied_id: str, expected_id: str) -> Task:
        if self._task is None:
            raise RuntimeError("reset first")
        if self.done:
            raise ValueError("episode already terminal")
        if len(self._events) >= self._task.max_actions:
            self.done = True
            self._save_trace()
            raise ValueError("action budget exhausted")
        if name not in self._task.required_tools or supplied_id != expected_id:
            self.reward -= 0.1
            self._record(name, {"supplied_id": supplied_id}, {"ok": False, "error": "wrong_tool_or_identifier"})
            raise ValueError("wrong tool or identifier for this task")
        return self._task

    def _record(self, name: str, arguments: dict, result: dict) -> str:
        previous_success = any(event["tool"] == name and event["result"].get("ok")
                               for event in self._events)
        previous = self._events[-1]["hash"] if self._events else "GENESIS"
        event = {"sequence": len(self._events) + 1, "tool": name, "arguments": arguments,
                 "result": result, "previous_hash": previous}
        event["hash"] = _digest(event)
        event["evidence_id"] = f"e{event['sequence']}-{event['hash'][:12]}"
        self._events.append(event)
        if result.get("ok"):
            self._evidence[event["evidence_id"]] = event
            self.reward += -0.04 if previous_success else 0.04
        self._save_trace()
        return event["evidence_id"]

    def _source_result(self, record: dict) -> dict:
        return {"source_status": "CACHED", "source_url": record["url"],
                "captured_at": record["captured_at"], "raw_sha256": record["raw_sha256"]}

    def lookup_uniprot(self, accession: str) -> str:
        """Retrieve a cached UniProt protein sequence window and provenance.

        Args:
            accession: Exact UniProt accession in the current task.
        """
        if self._task is None:
            raise RuntimeError("reset first")
        task = self._require("lookup_uniprot", accession, self._task.input_data["accession"])
        record = SOURCES[task.source_key]
        if task.fault_once and not self._fault_fired:
            self._fault_fired = True
            eid = self._record("lookup_uniprot", {"accession": accession},
                               {"ok": False, "error": "transient_upstream_timeout"})
            return json.dumps({"ok": False, "evidence_id": eid, "error": "transient_upstream_timeout"})
        result = {"ok": True, "accession": accession, "sequence_window": record["segment"],
                  "window_start": record["segment_start"], "protein_length": record["length"],
                  **self._source_result(record)}
        eid = self._record("lookup_uniprot", {"accession": accession}, result)
        return json.dumps({**result, "evidence_id": eid})

    def check_fasta(self, fasta: str) -> str:
        """Validate the supplied FASTA without silently changing it.

        Args:
            fasta: Exact FASTA text from the current task observation.
        """
        if self._task is None:
            raise RuntimeError("reset first")
        task = self._require("check_fasta", fasta, self._task.input_data["fasta"])
        lines = fasta.strip().splitlines()
        sequence = "".join(lines[1:]).upper() if lines and lines[0].startswith(">") else ""
        invalid = sorted(set(sequence) - AMINO_ACIDS)
        result = {"ok": True, "valid": bool(sequence) and not invalid,
                  "invalid_symbols": invalid, "sequence": sequence,
                  "source_status": "FIXTURE", "input_sha256": hashlib.sha256(fasta.encode()).hexdigest()}
        eid = self._record("check_fasta", {"fasta_sha256": result["input_sha256"]}, result)
        return json.dumps({**result, "evidence_id": eid})

    def lookup_rcsb(self, pdb_id: str) -> str:
        """Retrieve cached RCSB PDB experimental-method metadata and provenance.

        Args:
            pdb_id: Exact PDB identifier from the current task.
        """
        if self._task is None:
            raise RuntimeError("reset first")
        task = self._require("lookup_rcsb", pdb_id, self._task.input_data["pdb_id"])
        record = SOURCES[task.secondary_source_key or task.source_key]
        if task.fault_once and not self._fault_fired:
            self._fault_fired = True
            eid = self._record("lookup_rcsb", {"pdb_id": pdb_id},
                               {"ok": False, "error": "transient_upstream_timeout"})
            return json.dumps({"ok": False, "evidence_id": eid, "error": "transient_upstream_timeout"})
        result = {"ok": True, "pdb_id": pdb_id, "method": record["method"],
                  "title": record["title"], **self._source_result(record)}
        eid = self._record("lookup_rcsb", {"pdb_id": pdb_id}, result)
        return json.dumps({**result, "evidence_id": eid})

    def analyze_peptide(self, peptide: str) -> str:
        """Inspect a computational peptide score; it is not experimental evidence.

        Args:
            peptide: Exact candidate peptide from the current task.
        """
        if self._task is None:
            raise RuntimeError("reset first")
        task = self._require("analyze_peptide", peptide, self._task.input_data["peptide"])
        record = SOURCES[task.source_key]
        result = {"ok": True, "peptide": peptide, "standard_residues": set(peptide) <= AMINO_ACIDS,
                  "in_sequence_window": peptide in record["segment"],
                  "experimental_validation": False,
                  "method": "exact_substring_in_cached_window_v1", **self._source_result(record)}
        eid = self._record("analyze_peptide", {"peptide": peptide}, result)
        return json.dumps({**result, "evidence_id": eid})

    def submit_conclusion(self, claim: str, evidence_ids: str, source_status: str, limitation: str) -> str:
        """Submit a bounded scientific decision with tool evidence and provenance.

        Args:
            claim: Scientific finding label. Sequence labels: sequence_match,
                sequence_mismatch, invalid_fasta. Peptide labels:
                sequence_window_contains_peptide, sequence_window_lacks_peptide,
                insufficient_experimental_evidence. Structure labels: xray,
                nmr, electron_microscopy. Joint audits use sequence|method.
            evidence_ids: Comma-separated IDs returned by all required successful tools.
            source_status: Provenance of the external database record, usually CACHED.
            limitation: Scope label inferred from evidence: database_metadata_only for
                sequence/structure metadata, sequence_occurrence_not_validation for peptide
                evidence, or independent_metadata_only for a two-record audit.
        """
        if self._task is None or self.done:
            raise ValueError("no active episode")
        task = self._task
        if len(self._events) >= task.max_actions:
            self.done = True
            self._save_trace()
            raise ValueError("action budget exhausted")
        cited = {item.strip() for item in evidence_ids.split(",") if item.strip()}
        successful_tools = {self._evidence[eid]["tool"] for eid in cited if eid in self._evidence}
        all_evidence = cited <= self._evidence.keys() and set(task.required_tools) <= successful_tools
        provenance_ok = source_status == "CACHED" and limitation == task.limitation
        claim_ok = claim == task.expected_claim
        used_required_tools = set(task.required_tools) <= successful_tools
        redundant_or_invalid = any(event["result"].get("error") == "wrong_tool_or_identifier"
                                   for event in self._events)
        successful_names = [event["tool"] for event in self._events if event["result"].get("ok")]
        redundant_or_invalid |= len(successful_names) != len(set(successful_names))
        self.labops = (1.0 if used_required_tools and not redundant_or_invalid else
                       0.5 if used_required_tools else 0.0)
        self.researchsense = (1.0 if all_evidence and provenance_ok and claim_ok else
                              0.5 if all_evidence and provenance_ok else 0.0)
        self.success = bool(claim_ok and self.researchsense == 1.0)
        self.done = True
        self.reward += (1.0 if self.success else 0.0) + 0.1 * self.labops + 0.2 * self.researchsense
        if claim not in {"sequence_match", "sequence_mismatch", "invalid_fasta",
                         "sequence_window_contains_peptide", "sequence_window_lacks_peptide",
                         "insufficient_experimental_evidence", "xray", "nmr",
                         "electron_microscopy"} and "|" not in claim:
            self.reward -= 0.2
        finding = {"verified_success": self.success, "claim_correct": claim_ok,
                   "evidence_complete": all_evidence, "provenance_correct": provenance_ok,
                   "labops": self.labops, "researchsense": self.researchsense,
                   "reward": round(self.reward, 4)}
        self._record("submit_conclusion", {"claim": claim, "evidence_ids": sorted(cited),
                    "source_status": source_status, "limitation": limitation}, finding)
        return json.dumps(finding)

    def _save_trace(self) -> None:
        if self._task is None:
            return
        directory = os.getenv("BIOGYM_RL_TRACE_DIR")
        if not directory:
            return
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{self._attempt_id}.json"
        payload = {"schema_version": SCHEMA_VERSION, "scorer_version": SCORER_VERSION,
                   "suite_hash": SUITE_HASH, "attempt_id": self._attempt_id,
                   "case_id": self._task.id, "split": self._task.split,
                   "source_key": self._task.source_key, "events": self._events,
                   "reward": self.reward, "done": self.done, "success": self.success,
                   "labops": self.labops, "researchsense": self.researchsense}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(path)


def reward_outcome(environments, **kwargs) -> list[float]:
    return [1.0 if env.success else 0.0 for env in environments]


def reward_process(environments, **kwargs) -> list[float]:
    return [round(env.reward - (1.0 if env.success else 0.0), 4) for env in environments]
