from __future__ import annotations

import hashlib
import os
import random
import uuid
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from .cases import CASES
from .engine import EpisodeEngine
from .models import Action, SourceStatus
from .store import Store


TOOL_ACTIONS = ("validate_fasta", "inspect_structure", "analyze_epitope")
CLAIM_ACTIONS = (
    "valid_sequence", "invalid_sequence", "experimental_structure",
    "predicted_structure", "computational_candidate", "clinically_validated",
    "abstain",
)
ACTION_COUNT = len(TOOL_ACTIONS) + len(CLAIM_ACTIONS)
FEATURE_COUNT = 15


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 32), nn.Tanh(),
            nn.Linear(32, ACTION_COUNT),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


def training_device() -> torch.device:
    requested = os.getenv("BIOGYM_DEVICE", "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("BIOGYM_DEVICE must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("BIOGYM_DEVICE=cuda but PyTorch cannot access a CUDA GPU")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def features(observation: dict) -> torch.Tensor:
    data = observation["input_data"]
    result = (observation.get("last_result") or {}).get("result") or {}
    fasta = str(data.get("fasta", ""))
    sequence = "".join(fasta.splitlines()[1:]).upper()
    amino = set("ACDEFGHIKLMNPQRSTVWY")
    values = [
        float(observation["room"] == "sequence"),
        float(observation["room"] == "structure"),
        float(observation["room"] == "epitope"),
        float(observation["phase"] == "tool"),
        float(observation["phase"] == "conclude"),
        float(bool(sequence)),
        float(bool(set(sequence) - amino)),
        float(data.get("origin") == "experimental"),
        float(data.get("origin") == "predicted"),
        float(bool(data.get("peptide"))),
        float(result.get("valid") is True),
        float(result.get("valid") is False),
        float(result.get("origin") == "experimental"),
        float(result.get("origin") == "predicted"),
        float(result.get("experimental_validation") is False and bool(result)),
    ]
    return torch.tensor(values, dtype=torch.float32)


def _distribution(model: PolicyNet, observation: dict) -> Categorical:
    device = next(model.parameters()).device
    logits = model(features(observation).to(device))
    mask = torch.full_like(logits, -1e9)
    if observation["phase"] == "tool":
        mask[:len(TOOL_ACTIONS)] = 0
    else:
        mask[len(TOOL_ACTIONS):] = 0
    return Categorical(logits=logits + mask)


def _action(index: int, attempt_id: str, step: int, events: list, observation: dict) -> Action:
    action_id = f"policy-{attempt_id}-{step}"
    if index < len(TOOL_ACTIONS):
        return Action(action_id=action_id, type="tool_call", name=TOOL_ACTIONS[index])
    claim = CLAIM_ACTIONS[index - len(TOOL_ACTIONS)]
    if claim == "abstain":
        return Action(action_id=action_id, type="abstain")
    evidence = [
        event.event_id for event in events
        if event.kind == "tool_result" and (event.observation.get("last_result") or {}).get("ok")
    ]
    return Action(
        action_id=action_id, type="conclude", claim=claim,
        evidence_ids=evidence[-1:], source_status_claim=SourceStatus(observation["source_status"]),
    )


def rollout(
    engine: EpisodeEngine, model: PolicyNet, case_id: str, seed: int,
    policy_id: str, sample: bool,
) -> tuple[float, list[torch.Tensor], str, bool]:
    attempt = engine.reset(case_id, policy_id=policy_id, seed=seed)
    log_probs: list[torch.Tensor] = []
    total_reward = 0.0
    success = False
    for step in range(attempt.max_steps):
        distribution = _distribution(model, attempt.observation)
        choice = distribution.sample() if sample else distribution.logits.argmax()
        if sample:
            log_probs.append(distribution.log_prob(choice))
        action = _action(int(choice), attempt.id, step, attempt.events, attempt.observation)
        result = engine.step(attempt.id, action)
        total_reward += result.reward
        success = result.components["outcome"] == 1
        attempt = engine.state(attempt.id)
        if result.done:
            break
    return total_reward, log_probs, attempt.id, success


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_baseline(store: Store, artifact_dir: Path, seed: int = 7) -> dict:
    torch.manual_seed(seed)
    model = PolicyNet()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"baseline-{seed}.pt"
    torch.save(model.state_dict(), path)
    record = {
        "id": f"baseline-{seed}", "artifact_path": str(path), "sha256": _sha(path),
        "parent_id": None, "optimizer_steps": 0,
        "metadata": {"architecture": "small-policy-gradient-v1", "seed": seed,
                     "role": "starter trainable policy; not a language model"},
    }
    store.put_policy(record)
    return record


def load_policy(record: dict, device: torch.device | str = "cpu") -> PolicyNet:
    path = Path(record["artifact_path"])
    if _sha(path) != record["sha256"]:
        raise ValueError("policy artifact hash mismatch")
    model = PolicyNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.to(device).eval()
    return model


def train_policy(
    store: Store, engine: EpisodeEngine, artifact_dir: Path,
    seed: int = 7, episodes: int = 400,
) -> dict:
    if episodes < 1 or episodes > 20000:
        raise ValueError("episodes must be between 1 and 20000")
    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    device = training_device()
    baseline = create_baseline(store, artifact_dir, seed)
    model = load_policy(baseline, device=device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.006)
    train_ids = [case.id for case in CASES if case.split == "train"]
    history = []
    moving_baseline = 0.0
    for episode in range(episodes):
        case_id = train_ids[episode % len(train_ids)]
        reward, log_probs, _, _ = rollout(
            engine, model, case_id, seed + episode, baseline["id"], sample=True,
        )
        moving_baseline = 0.95 * moving_baseline + 0.05 * reward
        if log_probs:
            loss = -torch.stack(log_probs).sum() * (reward - moving_baseline)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if episode % max(1, episodes // 20) == 0 or episode == episodes - 1:
            history.append({"episode": episode + 1, "reward": round(reward, 4),
                            "moving_baseline": round(moving_baseline, 4)})
    policy_id = f"trained-{uuid.uuid4().hex[:10]}"
    path = artifact_dir / f"{policy_id}.pt"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    torch.save(model.state_dict(), path)
    record = {
        "id": policy_id, "artifact_path": str(path), "sha256": _sha(path),
        "parent_id": baseline["id"], "optimizer_steps": episodes,
        "metadata": {"architecture": "small-policy-gradient-v1", "algorithm": "REINFORCE",
                     "seed": seed, "train_case_ids": train_ids, "episodes": episodes,
                     "history": history, "device": str(device),
                     "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
                     "role": "starter trainable policy; not a language model"},
    }
    if record["sha256"] == baseline["sha256"]:
        raise RuntimeError("training produced no policy artifact change")
    store.put_policy(record)
    return {"baseline": baseline, "trained": record, "history": history}
