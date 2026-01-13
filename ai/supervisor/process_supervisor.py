from __future__ import annotations
import os, sys, time, json, yaml, logging, subprocess, threading
from pathlib import Path
from datetime import datetime

class RotatingPopen:
    def __init__(self, name: str, cmd: list[str], log_dir: Path, rotate_mb: int, keep_files: int, env: dict | None):
        self.name = name
        self.cmd = cmd
        self.log_dir = log_dir
        self.rotate_bytes = int(rotate_mb) * 1024 * 1024
        self.keep_files = keep_files
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None
        self._log_fp = None
        self._log_path = None
        self._bytes = 0
        self._lock = threading.Lock()

    def _open_log(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self._log_path = self.log_dir / f"{self.name}_{ts}.log"
        self._log_fp = open(self._log_path, "ab", buffering=0)
        self._bytes = 0

    def _rotate(self):
        if self._bytes < self.rotate_bytes:
            return
        try:
            self._log_fp.close()
        except Exception:
            pass
        files = sorted(self.log_dir.glob(f"{self.name}_*.log"))
        while len(files) >= self.keep_files:
            try:
                files[0].unlink()
                files.pop(0)
            except Exception:
                break
        self._open_log()

    def start(self):
        self._open_log()
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self.env,
            bufsize=1,
            universal_newlines=False,  # keep as bytes
        )
        t = threading.Thread(target=self._pump, daemon=True)
        t.start()

    def _pump(self):
        assert self.proc and self.proc.stdout
        for chunk in iter(lambda: self.proc.stdout.read(4096), b""):
            with self._lock:
                self._log_fp.write(chunk)
                self._bytes += len(chunk)
                self._rotate()

    def poll(self):
        return None if not self.proc else self.proc.poll()

    def terminate(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

    def kill(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass


class ProcessSpec:
    def __init__(self, name, cmd, restart, backoff_seconds, env):
        self.name = name
        self.cmd = cmd
        self.restart = restart
        self.backoff_seconds = int(backoff_seconds)
        self.env = env or {}

class ProcessSupervisor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.log_dir = Path(cfg.get("logging", {}).get("dir", "logs"))
        self.rotate_mb = cfg.get("logging", {}).get("rotate_mb", 20)
        self.keep_files = cfg.get("logging", {}).get("keep_files", 5)
        self.health_every = int(cfg.get("supervisor", {}).get("health_check_every", 10))
        self.kill_timeout = int(cfg.get("supervisor", {}).get("kill_timeout", 10))
        self.global_env = cfg.get("supervisor", {}).get("env", {}) or {}

        self.specs: list[ProcessSpec] = []
        # main executor
        ex = cfg.get("executor", {})
        self.specs.append(ProcessSpec("executor", ex["cmd"], ex.get("restart", "always"), ex.get("backoff_seconds", 5), ex.get("env", {})))
        for sc in cfg.get("sidecars", []):
            self.specs.append(ProcessSpec(sc["name"], sc["cmd"], sc.get("restart","always"), sc.get("backoff_seconds",5), sc.get("env", {})))

        self.runners: dict[str, RotatingPopen] = {}
        self.backoff_until: dict[str, float] = {}

        self.logger = logging.getLogger("LiveSupervisor")
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    def _env(self, spec: ProcessSpec):
        env = {**os.environ, **self.global_env, **spec.env}
        return env

    def _start(self, spec: ProcessSpec):
        now = time.time()
        until = self.backoff_until.get(spec.name, 0)
        if now < until:
            return  # still backing off

        self.logger.info(f"▶️ starting {spec.name}: {' '.join(spec.cmd)}")
        runner = RotatingPopen(
            name=spec.name, cmd=spec.cmd, log_dir=self.log_dir,
            rotate_mb=self.rotate_mb, keep_files=self.keep_files, env=self._env(spec)
        )
        runner.start()
        self.runners[spec.name] = runner

    def _maybe_restart(self, spec: ProcessSpec):
        runner = self.runners.get(spec.name)
        if not runner:
            self._start(spec)
            return
        rc = runner.poll()
        if rc is None:
            return  # still running

        self.logger.warning(f"⚠️ {spec.name} exited with code {rc}")
        if spec.restart == "never":
            return
        if spec.restart == "on-failure" and rc == 0:
            return
        # schedule restart after backoff
        self.backoff_until[spec.name] = time.time() + spec.backoff_seconds
        self._start(spec)

    def run_forever(self):
        # first start all
        for spec in self.specs:
            self._start(spec)
        try:
            while True:
                for spec in self.specs:
                    self._maybe_restart(spec)
                time.sleep(self.health_every)
        except KeyboardInterrupt:
            self.logger.info("⏹ stopping…")
            for r in self.runners.values():
                r.terminate()
            t0 = time.time()
            while time.time() - t0 < self.kill_timeout:
                if all(r.poll() is not None for r in self.runners.values()):
                    break
                time.sleep(0.5)
            for r in self.runners.values():
                r.kill()
