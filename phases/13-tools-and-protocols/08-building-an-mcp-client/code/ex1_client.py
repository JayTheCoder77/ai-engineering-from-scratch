#!/usr/bin/env python3
"""Exercise 1, step 2 — real subprocess spawn + EOF-death detection.

Spawns `server_proc.py` as a child process over stdio and runs a BACKGROUND
reader thread that drains its stdout. YOUR JOB (see TODO(human) below) is to
detect EOF and mark the session dead — that is the exact mechanism the docs
describe at docs/en.md:79 ("detect EOF on stdout ... treats the session dead").

Run from the lesson directory:
    python code/ex1_client.py
Then either:  kill -TERM <pid>   (pid is printed)  — or just wait; the script
self-terminates the child after a few seconds to demo the detection.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT = os.path.join(HERE, "server_proc.py")


@dataclass
class Session:
    name: str
    proc: subprocess.Popen
    alive: bool = True
    inbox: queue.Queue = None  # thread-safe line buffer, set in __post_init__

    def __post_init__(self) -> None:
        self.inbox = queue.Queue()


def _reader(session: Session) -> None:
    """Background thread: drain stdout, push lines to inbox, detect death.

    This runs for the whole life of the session. The ONLY thing you write
    is the EOF branch — everything else is wiring.
    """
    # --- TODO(human): implement EOF detection -------------------------
    # Loop forever, reading one line at a time from session.proc.stdout.
    #   line = session.proc.stdout.readline()
    # Two cases:
    #   * line == ""   -> END OF FILE. The child is gone (killed/crashed/
    #                     closed its stdout). This is the death signal.
    #                     Do: session.alive = False, print a message saying
    #                     which session died, and `return` to end the thread.
    #   * line != ""   -> a normal response line. Strip it and push it onto
    #                     session.inbox (session.inbox.put(line.strip())).
    # Hint: readline() BLOCKS until a line or EOF, so this thread naturally
    # waits. The single "if line == '':" check is the whole trick.
    # ----------------------------------------------------------------
    while session.alive:
        line = session.proc.stdout.readline()
        if line == "":
            session.alive =  False
            print(f"[death] Session : {session.name} stdout hit the eof")
            return 
        else:
            session.inbox.put(line.strip())


def main() -> None:
    proc = subprocess.Popen(
        [sys.executable, SERVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered so reads are per-line
    )
    session = Session(name="proc-server", proc=proc)
    threading.Thread(target=_reader, args=(session,), daemon=True).start()

    print(f"spawned {session.name}  pid={proc.pid}")
    print("to kill manually:   kill -TERM", proc.pid)

    # Prove the reader also delivers normal lines: send initialize, peek inbox.
    proc.stdin.write(json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}) + "\n")
    proc.stdin.flush()
    time.sleep(0.3)
    try:
        print("got from child:", session.inbox.get_nowait())
    except queue.Empty:
        print("no response yet")

    # Demo: if you don't kill it yourself, terminate after 3s to show EOF.
    deadline = time.time() + 6.0
    while session.alive and time.time() < deadline:
        time.sleep(0.2)
    if session.alive:
        print("auto-demo: terminating child to trigger EOF...")
        proc.terminate()

    # Wait for the reader thread to observe the death.
    while session.alive:
        time.sleep(0.1)
    print("MAIN: session.alive is now", session.alive, "-> exiting cleanly")


if __name__ == "__main__":
    main()
