"""Run with: python -m uvicorn biogym.openenv_server:app --port 8001."""
from __future__ import annotations

import os
from pathlib import Path

from openenv.core.env_server import create_app

from .openenv_env import BioGymAction, BioGymEnv, BioGymObservation, BioGymState
from .store import Store


DATA_DIR = Path(os.getenv("BIOGYM_DATA_DIR", Path(__file__).resolve().parents[1] / ".data"))
STORE = Store(DATA_DIR / "biogym.db")

app = create_app(
    env=lambda: BioGymEnv(STORE, allowed_splits={"train"}),
    action_cls=BioGymAction,
    observation_cls=BioGymObservation,
    state_cls=BioGymState,
    env_name="BioGym Arena",
    max_concurrent_envs=8,
)
