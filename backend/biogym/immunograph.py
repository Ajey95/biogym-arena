"""A pinned stdio MCP client for the released ImmunoGraph server."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


RELEASE = "0.2.1"
SERVER = (
    Path(__file__).resolve().parents[2] / "integrations" / "immunograph"
    / "node_modules" / "immunograph-mcp" / "dist" / "index.js"
)


async def call_tool(name: str, arguments: dict) -> dict:
    if name not in {"validate_sequence", "fetch_structure"}:
        raise ValueError("MCP tool is not authorized for this adapter")
    if not SERVER.exists():
        raise RuntimeError("ImmunoGraph release is not installed; run npm ci in integrations/immunograph")
    env = dict(os.environ)
    if name == "fetch_structure":
        env["RCSB_PDB_ENABLED"] = "true"
    params = StdioServerParameters(command="node", args=[str(SERVER)], env=env)
    async with asyncio.timeout(30):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.call_tool(name, arguments)
                payload = response.structuredContent
                if payload is None:
                    raise RuntimeError("MCP response has no structured content")
                return payload


def execute(case_id: str, name: str, input_data: dict, artifact_dir: Path) -> dict:
    if name == "immunograph_validate_sequence":
        payload = asyncio.run(call_tool("validate_sequence", {
            "fasta": input_data["fasta"], "profileVersion": "mvp-v1.0",
        }))
        if not payload.get("ok"):
            return {"ok": False, "error": payload.get("error"), "tool": name,
                    "mcp_meta": payload.get("meta"), "source_status": "FIXTURE"}
        data = payload["data"]
        return {
            "ok": True, "tool": name, "source_status": "FIXTURE",
            "execution_kind": "IMMUNOGRAPH_MCP_LOCAL_COMPUTATION",
            "mcp_release": RELEASE, "mcp_meta": payload["meta"],
            "result": {"valid": True, "length": data["sequenceLength"],
                       "sequence_sha256": data["sha256"], "warnings": data["warnings"]},
            "source_id": case_id,
        }
    if name == "immunograph_fetch_structure":
        accession = str(input_data["accession"]).upper()
        payload = asyncio.run(call_tool("fetch_structure", {
            "runId": f"biogym-{case_id}", "targetId": case_id,
            "source": "RCSB_PDB", "accession": accession,
            "fallbackPolicy": "LIVE_ONLY",
        }))
        if not payload.get("ok"):
            return {"ok": False, "error": payload.get("error"), "tool": name,
                    "mcp_meta": payload.get("meta"), "source_status": "UNKNOWN"}
        data = payload["data"]
        structure = data["structure"]
        provenance = data["provenance"]
        if structure.get("sourceStatus") != "LIVE" or provenance.get("status") != "LIVE":
            return {"ok": False, "error": "unexpected_source_status", "tool": name,
                    "source_status": str(structure.get("sourceStatus"))}
        artifact = structure["artifact"]
        raw = base64.b64decode(artifact["contentBase64"], validate=True)
        sha = hashlib.sha256(raw).hexdigest()
        if sha != artifact["sha256"] or len(raw) != artifact["byteLength"]:
            raise ValueError("ImmunoGraph PDB artifact integrity mismatch")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"{sha}.pdb"
        if not path.exists():
            path.write_bytes(raw)
        text = raw.decode("utf-8", errors="replace")
        method = next((line[10:].strip() for line in text.splitlines() if line.startswith("EXPDTA")), "not stated")
        if not any(line.startswith("ATOM") for line in text.splitlines()):
            raise ValueError("RCSB structure has no atomic coordinates")
        return {
            "ok": True, "tool": name, "source_status": "LIVE",
            "execution_kind": "IMMUNOGRAPH_MCP_RCSB_FETCH",
            "source_id": provenance.get("sourceUri"), "mcp_release": RELEASE,
            "mcp_meta": payload["meta"], "provenance": provenance,
            "artifact": {"sha256": sha, "byte_length": len(raw),
                         "media_type": "chemical/x-pdb", "url": f"/api/artifacts/{sha}.pdb"},
            "result": {"structure_id": accession, "chain_ids": structure["chainIds"],
                       "method": method, "origin": "experimental",
                       "has_coordinates": True},
        }
    raise ValueError("unknown ImmunoGraph adapter tool")
