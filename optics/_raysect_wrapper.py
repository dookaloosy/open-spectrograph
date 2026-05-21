"""Safe wrapper for raysect observer.observe() with worker-process cleanup.

Raysect's ``MulticoreEngine`` spawns worker processes via Python's
``multiprocessing`` module but has no try/finally in its ``run()``
method. If the parent process is interrupted (Ctrl+C, SIGTERM, unhandled
exception) while ``run()`` is blocked on ``result_queue.get()``, worker
processes are never terminated or joined. On WSL2 these orphaned Cython
workers saturate all cores and can hang the system hard enough to require
a power cycle.

This module provides:

- ``safe_observe(observer)`` — call ``observer.observe()`` with
  before/after child-process tracking; on any ``BaseException``,
  terminate+join new children before re-raising.
- A module-level ``atexit`` handler that kills all remaining child
  processes on interpreter shutdown (catches the case where the
  exception propagates past the caller without cleanup).
"""


import atexit
import multiprocessing
import os


def safe_observe(observer) -> None:
    """Call ``observer.observe()`` with robust worker cleanup on interrupt.

    Snapshots ``multiprocessing.active_children()`` before the call and,
    on any exception, terminates + joins every child process that appeared
    during the call. This covers ``KeyboardInterrupt`` (Ctrl+C),
    ``SystemExit``, and unexpected exceptions from raysect internals.
    """
    children_before = {p.pid for p in multiprocessing.active_children()}
    try:
        observer.observe()
    except BaseException:
        _kill_new_children(children_before)
        raise


def _kill_new_children(before_pids: set[int]) -> None:
    """SIGKILL child processes spawned since *before_pids*.

    Uses SIGKILL immediately (no SIGTERM+join) because raysect Cython
    workers hold the GIL and can't process SIGTERM promptly.
    """
    new = [p for p in multiprocessing.active_children() if p.pid not in before_pids]
    for proc in new:
        try:
            proc.kill()  # SIGKILL — uncatchable, no GIL needed
        except OSError:
            pass
    for proc in new:
        proc.join(timeout=3)


# ── Module-level safety net ──────────────────────────────────────────────
#
# If the exception from safe_observe propagates past the caller and the
# interpreter shuts down, atexit handlers still fire (for normal exit and
# unhandled exceptions — not SIGKILL, but that's OS-level and unavoidable).

def _atexit_cleanup() -> None:
    """SIGKILL any surviving child processes on interpreter shutdown."""
    for proc in multiprocessing.active_children():
        try:
            proc.kill()
        except OSError:
            pass
    for proc in multiprocessing.active_children():
        proc.join(timeout=2)


atexit.register(_atexit_cleanup)
