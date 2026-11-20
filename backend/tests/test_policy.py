from pathlib import Path

import torch

from biogym.engine import EpisodeEngine
from biogym.evaluation import evaluate
from biogym.policy import load_policy, train_policy
from biogym.store import Store


def test_training_changes_artifact_and_evaluation_is_paired(tmp_path):
    store = Store(tmp_path / "test.db")
    engine = EpisodeEngine(store)
    result = train_policy(store, engine, tmp_path / "policies", seed=4, episodes=18)
    assert result["trained"]["optimizer_steps"] == 18
    assert result["trained"]["metadata"]["device"] == ("cuda" if torch.cuda.is_available() else "cpu")
    assert result["baseline"]["sha256"] != result["trained"]["sha256"]
    assert load_policy(result["trained"]) is not None
    report = evaluate(store, engine, result["baseline"]["id"], result["trained"]["id"], repeats=1)
    assert report["case_ids"] == ["seq-03", "struct-03", "epi-03"]
    assert len(report["rows"]) == 6
    for case_id in report["case_ids"]:
        matched = [row for row in report["rows"] if row["case_id"] == case_id]
        assert len(matched) == 2
        assert matched[0]["seed"] == matched[1]["seed"]
        assert matched[0]["source_status"] == matched[1]["source_status"]
