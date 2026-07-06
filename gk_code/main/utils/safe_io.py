"""
Atomic, signal-safe wrappers for joblib.dump and Keras model.save.

Two layers of protection:
  1. Atomic write — write to a sibling .tmp file then os.replace() to the
     final path. os.replace() is atomic on the same filesystem, so the target
     is never left half-written even if the process is SIGKILLed mid-rename.
  2. Signal deferral — SIGTERM / SIGINT are caught and queued during the write
     so a SLURM scancel or Ctrl-C cannot interrupt an in-flight save.
     After the save completes the deferred signals are re-raised.

Usage:
    from safe_io import safe_joblib_dump, safe_keras_save

    safe_joblib_dump(obj, path, compress=3)   # drop-in for joblib.dump
    safe_keras_save(model, path)              # drop-in for model.save(path)
"""

import os
import signal
import shutil
import tempfile
from contextlib import contextmanager

import joblib


# ---------------------------------------------------------------------------
# Signal deferral
# ---------------------------------------------------------------------------

@contextmanager
def _defer_signals(*sigs):
    """Queue SIGTERM/SIGINT during the context, re-raise them on exit."""
    deferred = []
    old_handlers = {}

    def _capture(sig, frame):
        deferred.append((sig, frame))

    for sig in sigs:
        try:
            old_handlers[sig] = signal.signal(sig, _capture)
        except (OSError, ValueError):
            pass  # not in main thread — signals cannot be re-registered; skip

    try:
        yield
    finally:
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass
        for sig, _ in deferred:
            handler = old_handlers.get(sig)
            if callable(handler):
                handler(sig, None)
            else:
                os.kill(os.getpid(), sig)  # SIG_DFL / SIG_IGN — re-send to self


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def safe_joblib_dump(obj, path, compress=3):
    """
    Atomic + signal-safe replacement for joblib.dump(obj, path, compress=N).

    Writes to a sibling .tmp file in the same directory (guaranteeing the
    rename stays on the same filesystem), then atomically replaces the target.
    If the write fails the .tmp file is removed and the exception is re-raised.
    """
    path = str(path)
    dirn = os.path.dirname(os.path.abspath(path))
    os.makedirs(dirn, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirn, suffix=".tmp")
    os.close(fd)
    with _defer_signals(signal.SIGTERM, signal.SIGINT):
        try:
            joblib.dump(obj, tmp_path, compress=compress)
            os.replace(tmp_path, path)   # atomic on POSIX
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def safe_keras_save(model_or_layer, path):
    """
    Atomic + signal-safe replacement for model.save(path).

    Works for both .keras single-file saves and SavedModel directory saves.
    Saves to a sibling tmp location (same directory), then renames to path.
    If the save fails the tmp location is removed and the exception re-raised.
    """
    path = str(path)
    dirn = os.path.dirname(os.path.abspath(path))
    os.makedirs(dirn, exist_ok=True)
    tmp_path = os.path.join(dirn, f"_tmp_{os.path.basename(path)}")

    with _defer_signals(signal.SIGTERM, signal.SIGINT):
        try:
            model_or_layer.save(tmp_path)
            if os.path.isdir(tmp_path):
                # SavedModel directory — remove old target first, then rename
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.rename(tmp_path, path)
            else:
                os.replace(tmp_path, path)
        except Exception:
            try:
                if os.path.isdir(tmp_path):
                    shutil.rmtree(tmp_path)
                elif os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise
