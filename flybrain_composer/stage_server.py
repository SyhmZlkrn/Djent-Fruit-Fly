"""Static file server for stage/ that never lets the browser cache (modules change while iterating),
plus a tiny launcher API so the page's buttons can start the Python conductor themselves:

    python -m flybrain_composer.stage_server [port]

    GET /api/status                       -> {running, mode, pid, ws_up, since, exit, log[]}
    GET /api/launch?mode=play|improvise|generate -> starts `python -m flybrain_composer.cli play [--improvise|--generate]`
                 [&restart=1][&window=0]     (restart kills a conductor this server started first)
    GET /api/stop                         -> kills the conductor this server started (and its planner/workers)

The conductor itself serves stage/ with the same handler when port 8000 is free, so the page works
either way; launching/restarting from the page needs this standalone server (a process cannot
restart itself).
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from . import config

WS_PORT = 8765
LOG_PATH = config.OUTPUT_DIR / "conductor.log"

_LAUNCH: dict = {"proc": None, "mode": None, "started": 0.0, "args": []}
_LOCK = threading.Lock()
IS_CONDUCTOR = False          # set by player.StageServer: this process *is* a conductor


def ws_up(port: int = WS_PORT, timeout: float = 0.25) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _tail(path: Path, n: int = 8) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    keep = [ln for ln in lines if ln.strip() and "Warning" not in ln and "warn" not in ln.lower()]
    return keep[-n:]


def status() -> dict:
    proc = _LAUNCH["proc"]
    running = proc is not None and proc.poll() is None
    return {
        "running": running,
        "mode": _LAUNCH["mode"] if running else None,
        "pid": proc.pid if running else None,
        "since": round(time.time() - _LAUNCH["started"], 1) if running else None,
        "exit": (proc.returncode if (proc is not None and not running) else None),
        "ws_up": ws_up(),
        "is_conductor": IS_CONDUCTOR,
        "log": _tail(LOG_PATH) if (_LAUNCH["proc"] is not None) else [],
        "python": sys.executable,
    }


def _kill_tree(proc: subprocess.Popen):
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=10)
        else:
            os.killpg(os.getpgid(proc.pid), 15)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        proc.wait(5)
    except Exception:  # noqa: BLE001
        pass


def stop() -> dict:
    with _LOCK:
        _kill_tree(_LAUNCH["proc"])
        for _ in range(40):                      # wait for the websocket port to free up
            if not ws_up():
                break
            time.sleep(0.1)
        return {"ok": True, **status()}


def launch(mode: str = "play", restart: bool = False, window: bool = True, extra: list[str] | None = None) -> dict:
    """Start the conductor (`cli play`, optionally `--improvise`) as a detached child of this server."""
    if mode not in ("play", "improvise", "generate"):
        return {"ok": False, "hint": f"unknown mode {mode!r}"}
    if mode == "generate" and not (config.CACHE_DIR / "model_multi.npz").exists():
        return {"ok": False, "hint": "no multi-song model yet: put tabs in data/songs/ and run `python -m flybrain_composer.cli fit-multi`"}
    with _LOCK:
        st = status()
        if st["running"] and not restart:
            return {"ok": True, "already": True, **st}
        if st["running"] and restart:
            _kill_tree(_LAUNCH["proc"])
            for _ in range(40):
                if not ws_up():
                    break
                time.sleep(0.1)
        elif st["ws_up"]:
            if IS_CONDUCTOR:
                return {"ok": False, "self": True, **st,
                        "hint": "this page is served by the running conductor itself — start "
                                "`python -m flybrain_composer.stage_server` (or the 'stage' launch config) "
                                "to be able to launch or restart the conductor from here"}
            if not restart:
                return {"ok": True, "external": True, **st}
            return {"ok": False, **st, "hint": "a conductor that this server did not start is running — stop it first"}
        args = [sys.executable, "-W", "ignore", "-m", "flybrain_composer.cli", "play"]
        if mode == "improvise":
            args.append("--improvise")
        elif mode == "generate":
            args.append("--generate")
        if not window:
            args.append("--no-window")
        args += list(extra or [])
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log = open(LOG_PATH, "w", encoding="utf-8")
        kw: dict = {"cwd": str(config.ROOT), "stdout": log, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL}
        if os.name == "nt":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kw["start_new_session"] = True
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        kw["env"] = env
        try:
            proc = subprocess.Popen(args, **kw)
        except OSError as e:
            return {"ok": False, "hint": f"could not start the conductor: {e}"}
        _LAUNCH.update(proc=proc, mode=mode, started=time.time(), args=args)
        return {"ok": True, "started": True, "args": args, **status()}


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, *args, **kwargs):  # quiet
        pass

    def _json(self, payload: dict, code: int = 200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        try:
            if url.path == "/api/status":
                return self._json(status())
            if url.path == "/api/launch":
                extra = []
                if q.get("improv_off") in ("1", "true"):
                    extra.append("--improv-off")
                if q.get("fly_only") in ("1", "true"):
                    extra.append("--fly-only")
                if q.get("bpm"):
                    extra += ["--bpm", str(float(q["bpm"]))]
                if q.get("cycle"):
                    extra += ["--cycle", str(int(q["cycle"]))]
                if q.get("phrase_bars"):
                    extra += ["--phrase-bars", str(int(q["phrase_bars"]))]
                if q.get("wildness"):
                    extra += ["--wildness", str(float(q["wildness"]))]
                if q.get("snare") in ("24", "thirds"):
                    extra += ["--snare", q["snare"]]
                if q.get("density"):
                    extra += ["--density", str(float(q["density"]))]
                return self._json(launch(q.get("mode", "play"), restart=q.get("restart") in ("1", "true"),
                                         window=q.get("window", "1") not in ("0", "false"), extra=extra))
            if url.path == "/api/stop":
                return self._json(stop())
        except Exception as e:  # noqa: BLE001
            return self._json({"ok": False, "hint": f"{type(e).__name__}: {e}"}, 500)
        return self._json({"ok": False, "hint": "unknown api"}, 404)


def serve(port: int = 8000, directory=None, background: bool = False):
    handler = functools.partial(NoCacheHandler, directory=str(directory or config.STAGE_DIR))
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), handler)
    if background:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return httpd
    print(f"serving {directory or config.STAGE_DIR} on http://localhost:{port}/ (no-cache; /api/launch starts the conductor)", flush=True)
    try:
        httpd.serve_forever()
    finally:
        _kill_tree(_LAUNCH["proc"])


if __name__ == "__main__":
    serve(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
