import os
import signal
import shutil
import tempfile
from contextlib import contextmanager

import joblib


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
            pass
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
                os.kill(os.getpid(), sig) 

def safe_joblib_dump(obj, path, compress=3):
    path = str(path)
    dirn = os.path.dirname(os.path.abspath(path))
    os.makedirs(dirn, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dirn, suffix=".tmp")
    os.close(fd)
    with _defer_signals(signal.SIGTERM, signal.SIGINT):
        try:
            joblib.dump(obj, tmp_path, compress=compress)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def safe_keras_save(model_or_layer, path):
    path = str(path)
    dirn = os.path.dirname(os.path.abspath(path))
    os.makedirs(dirn, exist_ok=True)
    tmp_path = os.path.join(dirn, f"_tmp_{os.path.basename(path)}")

    with _defer_signals(signal.SIGTERM, signal.SIGINT):
        try:
            model_or_layer.save(tmp_path)
            if os.path.isdir(tmp_path):
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
