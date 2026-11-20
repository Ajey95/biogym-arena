from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class Store:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init(self) -> None:
        with self.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS attempts (
                id TEXT PRIMARY KEY, case_id TEXT NOT NULL, policy_id TEXT NOT NULL,
                seed INTEGER NOT NULL, status TEXT NOT NULL, step_count INTEGER NOT NULL,
                observation TEXT NOT NULL, created_at TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL, sequence INTEGER NOT NULL,
                action_id TEXT, payload TEXT NOT NULL, hash TEXT NOT NULL,
                UNIQUE(attempt_id, sequence), UNIQUE(attempt_id, action_id)
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS policies (
                id TEXT PRIMARY KEY, artifact_path TEXT NOT NULL, sha256 TEXT NOT NULL,
                parent_id TEXT, optimizer_steps INTEGER NOT NULL, metadata TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, result TEXT, error TEXT
            )""")
            db.execute("UPDATE jobs SET status='INTERRUPTED' WHERE status='RUNNING'")

    def insert_attempt(self, row: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (row["id"], row["case_id"], row["policy_id"], row["seed"], row["status"],
                 row["step_count"], json.dumps(row["observation"]), row["created_at"]),
            )

    def attempt(self, attempt_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM attempts WHERE id=?", (attempt_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["observation"] = json.loads(result["observation"])
        return result

    def events(self, attempt_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM events WHERE attempt_id=? ORDER BY sequence", (attempt_id,)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def action_event(self, attempt_id: str, action_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM events WHERE attempt_id=? AND action_id=?",
                (attempt_id, action_id),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def append(self, attempt_id: str, event: dict, status: str, observation: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (event["event_id"], attempt_id, event["sequence"], event.get("action_id"),
                 json.dumps(event), event["hash"]),
            )
            db.execute(
                "UPDATE attempts SET status=?, step_count=?, observation=? WHERE id=?",
                (status, event["sequence"], json.dumps(observation), attempt_id),
            )

    def put_policy(self, record: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO policies VALUES (?, ?, ?, ?, ?, ?)",
                       (record["id"], record["artifact_path"], record["sha256"],
                        record.get("parent_id"), record["optimizer_steps"], json.dumps(record["metadata"])))

    def policy(self, policy_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM policies WHERE id=?", (policy_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result["metadata"])
        return result

    def put_evaluation(self, evaluation_id: str, payload: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO evaluations VALUES (?, ?)", (evaluation_id, json.dumps(payload)))

    def evaluation(self, evaluation_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM evaluations WHERE id=?", (evaluation_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def latest_evaluation(self) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT payload FROM evaluations ORDER BY rowid DESC LIMIT 1").fetchone()
        return json.loads(row["payload"]) if row else None

    def put_job(self, job_id: str, kind: str, payload: dict) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES (?, ?, 'QUEUED', ?, NULL, NULL)",
                       (job_id, kind, json.dumps(payload)))

    def update_job(self, job_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?, result=?, error=? WHERE id=?",
                       (status, json.dumps(result) if result is not None else None, error, job_id))

    def job(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        for field in ("payload", "result"):
            if result[field] is not None:
                result[field] = json.loads(result[field])
        return result

    def latest_completed_job(self, kind: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT id FROM jobs WHERE kind=? AND status='COMPLETE' ORDER BY rowid DESC LIMIT 1",
                (kind,),
            ).fetchone()
        return self.job(row["id"]) if row else None
