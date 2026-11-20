"""Capture small, versioned UniProt/RCSB records for offline BioGym RL episodes.

The output holds normalized metadata and short sequence windows, not raw upstream
responses. Raw response hashes and capture times make later drift inspectable.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx


UNIPROT = {
    "train": ["P69905", "P68871", "P01308", "P02768", "P04637", "P60709", "P05067", "P00533"],
    "test": ["P01009", "P02787"],
    "transfer": ["P0DTC2", "P0DTC9"],
}
RCSB = {
    "train": ["4HHB", "1CRN", "1UBQ", "2PTC", "1VII", "1GFL", "1AKI", "7BV2"],
    "test": ["1L2Y", "2HYY"],
    "transfer": ["2POR", "1PYP"],
}


def _capture(client: httpx.Client, url: str) -> tuple[dict, str]:
    response = client.get(url, headers={"Accept": "application/json"})
    response.raise_for_status()
    raw = response.content
    return response.json(), hashlib.sha256(raw).hexdigest()


def build() -> dict:
    records: list[dict] = []
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for split, accessions in UNIPROT.items():
            for accession in accessions:
                url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
                data, raw_hash = _capture(client, url)
                sequence = data["sequence"]["value"]
                if len(sequence) < 32:
                    raise ValueError(f"{accession}: short sequence")
                records.append({
                    "kind": "uniprot", "id": accession, "split": split,
                    "url": url, "raw_sha256": raw_hash,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "source_status": "CACHED", "length": len(sequence),
                    "segment_start": 1, "segment": sequence[:32],
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "name": data.get("uniProtkbId", accession),
                })
        for split, accessions in RCSB.items():
            for accession in accessions:
                url = f"https://data.rcsb.org/rest/v1/core/entry/{accession}"
                data, raw_hash = _capture(client, url)
                methods = [item["method"] for item in data.get("exptl", [])]
                if not methods:
                    raise ValueError(f"{accession}: no experimental method")
                records.append({
                    "kind": "rcsb", "id": accession, "split": split,
                    "url": url, "raw_sha256": raw_hash,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "source_status": "CACHED", "method": methods[0],
                    "title": data.get("struct", {}).get("title", "")[:180],
                })
    return {"schema_version": "1.0.0", "sources": records,
            "interpretation": "Cached database metadata; no experimental or clinical validation by BioGym."}


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "backend" / "biogym" / "data" / "rl_snapshots_v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(target)
