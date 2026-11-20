"""OpenEnv 0.14 adapter over the same audited BioGym episode engine."""
from __future__ import annotations

import uuid
from typing import Literal

from openenv.core.env_server import Action as OpenEnvAction
from openenv.core.env_server import Environment, Observation, State
from pydantic import Field

from .cases import BY_ID
from .engine import EpisodeEngine
from .models import Action, SourceStatus
from .store import Store


class BioGymAction(OpenEnvAction):
    action_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: Literal["tool_call", "conclude", "abstain"]
    name: str | None = None
    claim: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source_status_claim: SourceStatus | None = None


class BioGymObservation(Observation):
    attempt_id: str
    case_id: str
    event_id: str
    data: dict


class BioGymState(State):
    case_id: str | None = None
    status: str = "READY"
    observation: dict = Field(default_factory=dict)
    event_count: int = 0


class BioGymEnv(Environment[BioGymAction, BioGymObservation, BioGymState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self, store: Store, allowed_splits: set[str] | None = None,
        policy_id: str = "openenv",
    ):
        self.engine = EpisodeEngine(store)
        self.allowed_splits = allowed_splits or {"train"}
        self.policy_id = policy_id
        self._attempt_id: str | None = None

    def reset(
        self, seed: int | None = None, episode_id: str | None = None,
        case_id: str = "seq-01", **kwargs,
    ) -> BioGymObservation:
        case = BY_ID.get(case_id)
        if case is None:
            raise KeyError(case_id)
        if case.split not in self.allowed_splits:
            raise ValueError(f"case split {case.split!r} is not enabled for this environment")
        attempt = self.engine.reset(case_id, policy_id=self.policy_id, seed=seed or 0)
        self._attempt_id = attempt.id
        return BioGymObservation(
            attempt_id=attempt.id, case_id=case_id,
            event_id=attempt.events[-1].event_id,
            data=attempt.observation, reward=0.0, done=False,
            metadata={"case_version": case.version, "split": case.split,
                      "external_episode_id": episode_id},
        )

    def step(self, action: BioGymAction, timeout_s: float | None = None, **kwargs) -> BioGymObservation:
        if self._attempt_id is None:
            raise RuntimeError("reset must be called before step")
        result = self.engine.step(self._attempt_id, Action.model_validate(
            action.model_dump(exclude={"metadata"})
        ))
        attempt = self.engine.state(self._attempt_id)
        return BioGymObservation(
            attempt_id=attempt.id, case_id=attempt.case_id,
            event_id=result.event_id, data=result.observation,
            reward=result.reward, done=result.done,
            metadata={"labops": result.components["labops"],
                      "researchsense": result.components["researchsense"],
                      "outcome": result.components["outcome"],
                      "terminal_reason": result.terminal_reason,
                      "remaining_steps": result.remaining_steps},
        )

    @property
    def state(self) -> BioGymState:
        if self._attempt_id is None:
            return BioGymState()
        attempt = self.engine.state(self._attempt_id)
        return BioGymState(
            episode_id=attempt.id, step_count=attempt.step_count,
            case_id=attempt.case_id, status=attempt.status,
            observation=attempt.observation, event_count=len(attempt.events),
        )
