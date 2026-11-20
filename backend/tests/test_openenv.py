import pytest

from biogym.openenv_env import BioGymAction, BioGymEnv
from biogym.store import Store


def test_openenv_adapter_uses_audited_episode_and_split_gate(tmp_path):
    env = BioGymEnv(Store(tmp_path / "env.db"))
    first = env.reset(case_id="seq-01", seed=12)
    assert first.data["phase"] == "tool"
    assert env.state.episode_id == first.attempt_id

    tool = env.step(BioGymAction(type="tool_call", name="validate_fasta"))
    assert tool.data["phase"] == "conclude"
    assert tool.reward is not None
    assert env.state.step_count == 1

    with pytest.raises(ValueError, match="not enabled"):
        env.reset(case_id="seq-03")
