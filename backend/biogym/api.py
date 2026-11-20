from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from .cases import BY_ID, CASES, ROOMS, public_case
from .engine import EpisodeEngine
from .evaluation import evaluate
from .export import build_attempt_export, build_evaluation_export
from .models import Action
from .policy import train_policy
from .store import Store
from .tools import fetch_uniprot


DATA_DIR = Path(os.getenv("BIOGYM_DATA_DIR", Path(__file__).resolve().parents[1] / ".data"))
STORE = Store(DATA_DIR / "biogym.db")
ENGINE = EpisodeEngine(STORE)
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="biogym-training")

app = FastAPI(title="BioGym Arena API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


class StartRequest(BaseModel):
    case_id: str
    policy_id: str = "manual"
    seed: int = 0


class TrainRequest(BaseModel):
    seed: int = 7
    episodes: int = Field(default=400, ge=1, le=20000)


class EvaluationRequest(BaseModel):
    baseline_policy_id: str
    trained_policy_id: str
    seed: int = 1000
    repeats: int = Field(default=3, ge=1, le=30)


def _not_found(error: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=f"Not found: {error}")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": "0.1.0", "science_status": "starter FIXTURE cases"}


@app.get("/api/rooms")
def rooms() -> list[dict]:
    return [
        {"id": room_id, **room, "cases": [public_case(case) for case in CASES if case.room == room_id]}
        for room_id, room in ROOMS.items()
    ]


@app.get("/api/cases")
def cases() -> list[dict]:
    return [public_case(case) for case in CASES]


@app.post("/api/attempts")
def start(request: StartRequest) -> dict:
    try:
        return ENGINE.reset(request.case_id, request.policy_id, request.seed).model_dump(mode="json")
    except KeyError as error:
        raise _not_found(error) from error


@app.get("/api/attempts/{attempt_id}")
def attempt(attempt_id: str) -> dict:
    try:
        return ENGINE.state(attempt_id).model_dump(mode="json")
    except KeyError as error:
        raise _not_found(error) from error


@app.post("/api/attempts/{attempt_id}/actions")
def step(attempt_id: str, action: Action) -> dict:
    try:
        return ENGINE.step(attempt_id, action).model_dump(mode="json")
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _train_job(job_id: str, request: TrainRequest) -> None:
    STORE.update_job(job_id, "RUNNING")
    try:
        result = train_policy(STORE, ENGINE, DATA_DIR / "policies", request.seed, request.episodes)
        STORE.update_job(job_id, "COMPLETE", result=result)
    except Exception as error:
        STORE.update_job(job_id, "FAILED", error=f"{type(error).__name__}: {error}")


@app.post("/api/train", status_code=202)
def train(request: TrainRequest) -> dict:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    STORE.put_job(job_id, "train", request.model_dump())
    EXECUTOR.submit(_train_job, job_id, request)
    return {"job_id": job_id, "status": "QUEUED"}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    result = STORE.job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@app.get("/api/training/latest")
def latest_training() -> dict | None:
    latest = STORE.latest_completed_job("train")
    return latest["result"] if latest else None


@app.post("/api/evaluations")
def run_evaluation(request: EvaluationRequest) -> dict:
    try:
        return evaluate(
            STORE, ENGINE, request.baseline_policy_id, request.trained_policy_id,
            request.seed, request.repeats,
        )
    except KeyError as error:
        raise _not_found(error) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/evaluations/{evaluation_id}")
def evaluation(evaluation_id: str) -> dict:
    report = STORE.evaluation(evaluation_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return report


@app.get("/api/evaluation/latest")
def latest_evaluation() -> dict | None:
    return STORE.latest_evaluation()


@app.get("/api/attempts/{attempt_id}/export")
def export_attempt(attempt_id: str) -> Response:
    try:
        data = build_attempt_export(STORE, attempt_id)
    except KeyError as error:
        raise _not_found(error) from error
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="biogym-attempt-{attempt_id}.zip"'
    })


@app.get("/api/evaluations/{evaluation_id}/export")
def export_evaluation(evaluation_id: str) -> Response:
    try:
        data = build_evaluation_export(STORE, evaluation_id)
    except KeyError as error:
        raise _not_found(error) from error
    return Response(data, media_type="application/zip", headers={
        "Content-Disposition": f'attachment; filename="biogym-evaluation-{evaluation_id}.zip"'
    })


@app.get("/api/sources/uniprot/{accession}")
async def live_uniprot(accession: str) -> dict:
    try:
        return await fetch_uniprot(accession)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Upstream unavailable: {type(error).__name__}") from error


@app.get("/api/artifacts/{digest}.pdb")
def pdb_artifact(digest: str) -> Response:
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise HTTPException(status_code=422, detail="Invalid artifact digest")
    path = DATA_DIR / "artifacts" / (digest + ".pdb")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return Response(path.read_bytes(), media_type="chemical/x-pdb")
