#!/usr/bin/env python3
"""Exercise 3 - exponential backoff on server restart.

A crashed MCP server shouldn't be respawned in a tight loop. restart()
must wait longer each failure (1s, 2s, 4s, ...), never exceed CAP seconds,
and notify the user after 3 consecutive failures. On success the failure
count resets.

Run: python code/ex3_backoff.py
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

BASE = 1      # initial delay (seconds)
CAP = 30      # maximum delay (seconds) - the exercise's "cap at 30"
NOTIFY_AFTER = 3

# Simulated flakiness: this server "crashes" on its first 3 start attempts,
# then stays up. Lets you watch the backoff without real process crashes.
FAIL_BEFORE_SUCCESS = 3


@dataclass
class Session:
    name: str
    failures: int = 0
    attempts: int = 0
    alive: bool = False

    def spawn(self) -> bool:
        """Attempt to (re)start the server. Returns True on success.

        In a real client this is subprocess.Popen(...) + re-handshake +
        restarting the reader thread. The backoff loop below is identical
        regardless of what spawn() actually does.
        """
        self.attempts += 1
        ok = self.attempts > FAIL_BEFORE_SUCCESS
        self.alive = ok
        if ok:
            print(f"  [{self.name}] spawn succeeded on attempt {self.attempts}")
        else:
            print(f"  [{self.name}] spawn FAILED (attempt {self.attempts})")
        return ok

    def restart(self) -> None:
        # === TODO(human): exponential backoff restart loop =============
        # Goal: keep trying spawn() until it succeeds.
        #   * On success: self.failures = 0 (reset!), return.
        #   * On failure: self.failures += 1, then:
        #       delay = min(BASE * 2 ** (self.failures - 1), CAP)
        #       if self.failures >= NOTIFY_AFTER: notify_user(...)
        #       print the delay, then time.sleep(delay), then loop again.
        # The `min(..., CAP)` is what enforces the 30s cap.
        # ==============================================================
        while not self.alive:
            # successful
            if self.spawn():
                self.failures = 0
                return
            
            self.failures += 1
            delay = min(BASE * 2 ** (self.failures - 1) , CAP) 
            if self.failures >= NOTIFY_AFTER:
                notify_user(self.name, self.failures)
            print(f"  delaying for {delay}s")
            time.sleep(delay)
        


def notify_user(server_name: str, failures: int) -> None:
    print(f"  ! NOTIFY: server '{server_name}' failed {failures} times in a "
          f"row - check its config / dependencies before retrying.")


def main() -> None:
    s = Session(name="proc-server")
    print("restarting with exponential backoff...")
    s.restart()
    print(f"done. alive={s.alive}  total_failures_reset={s.failures}")


if __name__ == "__main__":
    main()
