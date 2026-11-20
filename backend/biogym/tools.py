from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

import httpx

from .models import Case, SourceStatus


TOOL_NAMES = (
    "validate_fasta", "inspect_structure", "analyze_epitope",
    "immunograph_validate_sequence", "immunograph_fetch_structure",
)
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_fixture_tool(case: Case, name: str) -> dict:
    if name not in TOOL_NAMES:
        return {"ok": False, "error": "unknown_tool"}
    if name == "validate_fasta":
        fasta = str(case.input_data.get("fasta", ""))
        lines = [line.strip() for line in fasta.splitlines() if line.strip()]
        sequence = "".join(lines[1:]).upper() if lines and lines[0].startswith(">") else ""
        invalid = sorted(set(sequence) - AMINO_ACIDS)
        valid = bool(sequence) and not invalid and len(lines) >= 2
        result = {"valid": valid, "length": len(sequence), "invalid_symbols": invalid}
    elif name == "inspect_structure":
        data = case.input_data
        result = {"structure_id": data.get("structure_id"), "chain": data.get("chain"),
                  "method": data.get("method"), "origin": data.get("origin"),
                  "has_coordinates": False}
    else:
        data = case.input_data
        peptide = str(data.get("peptide", ""))
        result = {"peptide": peptide, "length": len(peptide),
                  "method": data.get("method"), "score": data.get("score"),
                  "experimental_validation": bool(data.get("experimental_validation", False)),
                  "interpretation": "computational prediction only"}
    return {
        "ok": True, "tool": name, "result": result,
        "source_status": case.source_status.value,
        "source_id": case.id, "method_version": "starter-fixture-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "input_hash": digest(case.input_data), "output_hash": digest(result),
    }


def run_scientific_tool(case: Case, name: str, artifact_dir) -> dict:
    if name.startswith("immunograph_"):
        from .immunograph import execute
        return execute(case.id, name, case.input_data, artifact_dir)
    return run_fixture_tool(case, name)


async def fetch_uniprot(accession: str) -> dict:
    """Optional live probe; its output is never used as a fixture score."""
    if not re.fullmatch(r"[A-Z0-9]{6,10}", accession):
        raise ValueError("Invalid UniProt accession")
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(url, headers={"Accept": "application/json"})
        response.raise_for_status()
        data = response.json()
    summary = {
        "accession": data.get("primaryAccession"),
        "name": data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value"),
        "length": data.get("sequence", {}).get("length"),
    }
    return {"ok": True, "source_status": SourceStatus.LIVE.value, "source_id": url,
            "captured_at": datetime.now(timezone.utc).isoformat(), "result": summary,
            "output_hash": digest(data), "method_version": "uniprot-rest-json"}
