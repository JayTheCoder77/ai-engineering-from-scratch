"""Phase 13 Lesson 13 - MCP async Tasks (SEP-1686) with durable state.

Simulates a long-running generate_report tool:
  - tools/call with _meta.task.required returns immediately with taskId
  - worker thread updates progress in a filesystem-backed task store
  - tasks/status polls progress
  - tasks/result returns the final payload
  - tasks/cancel signals the worker to stop
  - crash recovery marks in-flight tasks as failed on reload

Stdlib only.

Run: python code/main.py
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sqlite3

STORE_DIR = Path("/tmp/lesson-13-tasks")
STORE_DIR.mkdir(parents=True, exist_ok=True)

# db
con = sqlite3.connect("/home/jayant/projects/ai-engineering-from-scratch/phases/13-tools-and-protocols/13-mcp-async-tasks/code/tutorial.db" , check_same_thread=False)
cur = con.cursor()

cur.execute(""" 
    CREATE TABLE IF NOT EXISTS tasks(
        id TEXT PRIMARY KEY,
        state TEXT NOT NULL,
        progress FLOAT NOT NULL,
        total_ms INTEGER NOT NULL,
        result TEXT,
        error TEXT,
        ttl_ms INTEGER NOT NULL,
        created_at FLOAT NOT NULL,
        cancel_requested BOOLEAN NOT NULL
    )
""")
con.commit()


@dataclass
class Task:
    id: str
    state: str = "working"
    progress: float = 0.0
    total_ms: int = 0
    result: dict | None = None
    error: str | None = None
    ttl_ms: int = 900_000
    created_at: float = field(default_factory=time.time)
    cancel_requested: bool = False

    def persist(self) -> None:
        # (STORE_DIR / f"{self.id}.json").write_text(json.dumps(asdict(self), indent=2))
        cur.execute("""
            INSERT INTO tasks (id, state, progress, total_ms, result, error, ttl_ms, created_at, cancel_requested) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                state = excluded.state,
                progress = excluded.progress,
                total_ms = excluded.total_ms,
                result = excluded.result,
                error = excluded.error,
                ttl_ms = excluded.ttl_ms,
                created_at = excluded.created_at,
                cancel_requested = excluded.cancel_requested
        """, (self.id, self.state, self.progress, self.total_ms, json.dumps(self.result), self.error, self.ttl_ms, self.created_at, self.cancel_requested))
        con.commit()

    @classmethod
    def load(cls, tid: str) -> "Task | None":
        # p = STORE_DIR / f"{tid}.json"
        p = cur.execute("""
            SELECT 
                id,
                state,
                progress,
                total_ms,
                result,
                error,
                ttl_ms,
                created_at,
                cancel_requested
            FROM tasks WHERE id = ?
        """, (tid,)).fetchone()

        if not p:
            return None
        
        return cls(
            id=p[0],
            state=p[1],
            progress=p[2],
            total_ms=p[3],
            result=json.loads(p[4]) if p[4] else None,
            error=p[5],
            ttl_ms=p[6],
            created_at=p[7],
            cancel_requested=bool(p[8])
        )


class TaskStore:
    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.crash_recover()

    def crash_recover(self) -> None:
        # for p in STORE_DIR.glob("*.json"):
        cur.execute("SELECT id FROM tasks WHERE state = ?", ("working",))
        rows = cur.fetchall()
        for p in rows:
            tid = p[0]
            t = Task.load(tid)
            if t is None:
                continue
            if t.state == "working":
                t.state = "failed"
                t.error = "CRASH_RECOVERY"
                t.persist()
            self.tasks[t.id] = t

    def create(self, total_ms: int) -> Task:
        t = Task(id=f"tsk_{uuid.uuid4().hex[:12]}", total_ms=total_ms)
        t.persist()
        self.tasks[t.id] = t
        return t

    def update(self, tid: str, **changes) -> None:
        t = self.tasks[tid]
        for k, v in changes.items():
            setattr(t, k, v)
        t.persist()


STORE = TaskStore()


def worker_generate_report(task: Task, size: str) -> None:
    """Simulated 3-second report generation."""
    try:
        for step in range(30):
            if task.cancel_requested:
                STORE.update(task.id, state="cancelled")
                return
            time.sleep(0.1)
            STORE.update(task.id, progress=(step + 1) / 30)
        STORE.update(task.id, state="completed",
                     result={"content": [{"type": "text",
                                          "text": f"Report size={size} with 30 sections"}],
                             "isError": False})
    except Exception as e:
        STORE.update(task.id, state="failed", error=str(e))


def tools_call(name: str, args: dict, meta: dict | None = None) -> dict:
    if name != "generate_report":
        return {"isError": True,
                "content": [{"type": "text", "text": f"unknown tool {name}"}]}
    task_required = meta and meta.get("task", {}).get("required", False)
    if not task_required:
        # synchronous fallback path (could also be forbidden by the server)
        time.sleep(3.0)
        return {"isError": False,
                "content": [{"type": "text", "text": "Report generated synchronously"}]}
    task = STORE.create(total_ms=3000)
    threading.Thread(target=worker_generate_report,
                     args=(task, args.get("size", "medium")), daemon=True).start()
    return {"_meta": {"task": {"id": task.id, "state": task.state, "ttl": task.ttl_ms}}}


def tasks_status(tid: str) -> dict:
    t = STORE.tasks.get(tid)
    if not t:
        return {"error": "not found"}
    return {"taskId": tid, "state": t.state, "progress": round(t.progress, 2)}


def tasks_result(tid: str) -> dict:
    t = STORE.tasks.get(tid)
    if not t:
        return {"error": "not found"}
    if t.state != "completed":
        return {"error": f"not ready; state={t.state}"}
    return t.result or {}


def tasks_cancel(tid: str) -> dict:
    t = STORE.tasks.get(tid)
    if not t or t.state in {"completed", "failed", "cancelled"}:
        return {"taskId": tid, "state": t.state if t else "unknown"}
    STORE.update(tid, cancel_requested=True)
    return {"taskId": tid, "state": "cancelling"}


def demo() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 13 - MCP ASYNC TASKS (SEP-1686)")
    print("=" * 72)

    print("\n--- kick off generate_report as task ---")
    resp = tools_call("generate_report", {"size": "large"},
                      meta={"task": {"required": True}})
    tid = resp["_meta"]["task"]["id"]
    print(f"  task id: {tid}  state: {resp['_meta']['task']['state']}  "
          f"ttl: {resp['_meta']['task']['ttl']} ms")

    print("\n--- poll status until terminal ---")
    while True:
        status = tasks_status(tid)
        print(f"  state={status['state']:10s}  progress={status['progress']:.2f}")
        # result = tasks_result(tid)
        # print(f"  result: {result}")
        if status["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.5)

    print("\n--- fetch result ---")
    result = tasks_result(tid)
    print(f"  result: {result['content'][0]['text']}")

    print("\n--- cancellation demo ---")
    resp = tools_call("generate_report", {"size": "small"},
                      meta={"task": {"required": True}})
    tid2 = resp["_meta"]["task"]["id"]
    print(f"  spawned task {tid2}")
    time.sleep(0.4)
    cancel = tasks_cancel(tid2)
    print(f"  cancel request: {cancel}")
    while True:
        status = tasks_status(tid2)
        if status["state"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.3)
    print(f"  final state: {status}")

    print("\n--- crash recovery simulation ---")
    # write a fake task that claims to be working but has no worker
    fake = STORE.create(total_ms=1000)
    del STORE.tasks[fake.id]  # pretend process died
    # reload from disk
    store2 = TaskStore()
    recovered = store2.tasks.get(fake.id)
    print(f"  reloaded {fake.id} -> state={recovered.state}  error={recovered.error}")


if __name__ == "__main__":
    demo()
