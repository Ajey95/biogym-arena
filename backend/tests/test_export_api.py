import json
import zipfile
from io import BytesIO

from fastapi.testclient import TestClient

from biogym.api import app
from biogym.engine import EpisodeEngine
from biogym.export import build_attempt_export, verify_zip
from biogym.models import Action, SourceStatus
from biogym.store import Store


def test_export_contains_verifiable_event_chain(tmp_path):
    store = Store(tmp_path / "test.db")
    engine = EpisodeEngine(store)
    attempt = engine.reset("seq-01")
    tool = engine.step(attempt.id, Action(action_id="a", type="tool_call", name="validate_fasta"))
    engine.step(attempt.id, Action(
        action_id="b", type="conclude", claim="valid_sequence",
        evidence_ids=[tool.event_id], source_status_claim=SourceStatus.FIXTURE,
    ))
    data = build_attempt_export(store, attempt.id)
    manifest = verify_zip(data)
    assert manifest["offline_replay_verified"]
    with zipfile.ZipFile(BytesIO(data)) as archive:
        assert json.loads(archive.read("case.json"))["source_status"] == "FIXTURE"
        assert len(json.loads(archive.read("events.json"))) == 3


def test_api_exposes_rooms_and_rejects_unknown_case():
    client = TestClient(app)
    assert len(client.get("/api/rooms").json()) == 3
    assert client.post("/api/attempts", json={"case_id": "missing"}).status_code == 404
