from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SourceStatus(StrEnum):
    LIVE = "LIVE"
    CACHED = "CACHED"
    FIXTURE = "FIXTURE"
    SYNTHETIC = "SYNTHETIC"


class Action(BaseModel):
    action_id: str = Field(min_length=1, max_length=100)
    type: Literal["tool_call", "conclude", "abstain"]
    name: str | None = None
    claim: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source_status_claim: SourceStatus | None = None


class Case(BaseModel):
    id: str
    room: str
    mode: str
    split: Literal["train", "test", "transfer", "integration", "transfer_live"]
    prompt: str
    input_data: dict
    required_tool: str
    expected_claim: str
    source_status: SourceStatus = SourceStatus.FIXTURE
    fault_once: bool = False
    max_steps: int = 5
    version: str = "1.0.0"


class Event(BaseModel):
    event_id: str
    attempt_id: str
    sequence: int
    kind: str
    action_id: str | None = None
    action: dict | None = None
    observation: dict
    reward: float
    labops: float
    researchsense: float
    outcome: float
    previous_hash: str
    hash: str
    created_at: str


class StepResult(BaseModel):
    attempt_id: str
    event_id: str
    observation: dict
    reward: float
    components: dict[str, float]
    done: bool
    terminal_reason: str | None
    remaining_steps: int


class AttemptState(BaseModel):
    id: str
    case_id: str
    policy_id: str
    seed: int
    status: str
    step_count: int
    max_steps: int
    observation: dict
    events: list[Event]
