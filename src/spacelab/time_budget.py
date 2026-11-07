"""Persistent wall-clock deadline shared by notebooks, plus subprocess watchdog.

No reset API: a session's deadline is immutable. Notebook idle time counts too.
All expensive work (including splits/benchmark/final inference) must use launch().
"""
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from filelock import FileLock

from .io import read_json, write_json


class DeadlineExceeded(RuntimeError):
    pass


class Deadline:
    def __init__(self, stop_at):
        self.stop_at = float(stop_at)

    def remaining(self):
        return max(0., self.stop_at - time.time())

    def check(self, reserve=0):
        if self.remaining() <= reserve:
            raise DeadlineExceeded("Tiempo agotado; no se inicia otra operación.")


def _interrupt_manifests(run, reason):
    if not run:
        return
    for path in Path(run).glob("**/experiment_manifest.json"):
        manifest = read_json(path)
        if manifest.get("status") == "ejecutándose":
            manifest.update(status="interrumpido",reason=reason)
            write_json(path,manifest)


def launch(action, payload, storage, cap_seconds, *, final=False, estimate_seconds=None, unlimited=False):
    """Serial supervisor. Kills the worker at the allocated hard deadline.

    cap_seconds includes model load, downloads, training, scoring and saving.
    Never dispatch a configuration that lacks enough measured time to finish.
    """
    storage = Path(storage).expanduser().resolve()
    storage.mkdir(parents=True, exist_ok=True)
    with FileLock(str(storage / "budget.lock"), timeout=0):
        ledger_path = storage / "budget.json"
        state = read_json(ledger_path) if ledger_path.exists() else {
            "started_at": time.time(), "max_seconds": None if unlimited else 5400,
            "final_reserve_seconds": 480, "safety_seconds": 120, "jobs": []}
        # Reconcile a supervisor that disappeared. Wall time already counts;
        # gpu.lock still prevents overlap with a surviving worker.
        for previous in state["jobs"]:
            if previous["status"] == "ejecutándose":
                previous.update(status="interrumpido",reason="Supervisor anterior desconectado")
                _interrupt_manifests(previous.get("run"),previous["reason"])
        if unlimited != (state["max_seconds"] is None):
            raise ValueError("Use otra carpeta de sesión para cambiar el límite de tiempo.")
        if action == "benchmark" and state["max_seconds"] is not None:
            measured_used = sum(j.get("elapsed_seconds", 0) for j in state["jobs"] if j["action"]=="benchmark")
            cap_seconds = min(cap_seconds,max(0,300-measured_used))
        now = time.time()
        hard_end = (state["started_at"] + state["max_seconds"] - state["safety_seconds"]
                    if state["max_seconds"] is not None else float("inf"))
        usable = hard_end - now - (0 if final else state["final_reserve_seconds"])
        policy_path = storage / "runtime_deadline.json"
        if policy_path.exists():
            policy = read_json(policy_path)
            reserve = 0 if final else policy.get("final_reserve_seconds", 900)
            usable = min(usable, float(policy["stop_at"]) - now - reserve - 30)
        slot = min(float(cap_seconds), usable)
        job_id = f"{action}-{uuid.uuid4().hex[:12]}"
        job = {"id": job_id, "action": action, "status": "pendiente", "created_at": now,
               "cap_seconds": cap_seconds, "estimated_seconds": estimate_seconds}
        state["jobs"].append(job)
        if slot <= 15 or (estimate_seconds is not None and estimate_seconds > slot - 15):
            job.update(status="omitido", reason="Presupuesto insuficiente para la medición disponible")
            write_json(ledger_path, state)
            return job
        run = storage / "jobs" / job_id
        run.mkdir(parents=True, exist_ok=False)
        request = {"action": action, "payload": payload, "storage": str(storage),
                   "run": str(run), "stop_at": now + slot - 10}
        write_json(run / "request.json", request)
        job.update(status="ejecutándose", started_at=now, run=str(run))
        write_json(ledger_path, state)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
        process = None
        try:
            with (run / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-m", "spacelab.worker", str(run/"request.json")],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
                try:
                    code = process.wait(timeout=max(1, slot))
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       capture_output=True, check=False)
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code = -9
            result = read_json(run / "result.json") if (run/"result.json").exists() else {}
            job.update(status=result.get("status", "interrumpido"), returncode=code,
                       reason=result.get("reason"), result=result)
            if code != 0:
                job["status"] = "interrumpido"
        except BaseException:
            if process is not None and process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, check=False)
                else:
                    os.killpg(process.pid, signal.SIGKILL)
            job.update(status="interrumpido", reason="Supervisor interrumpido")
            raise
        finally:
            if job["status"] == "interrumpido":
                _interrupt_manifests(run,job.get("reason") or "Watchdog o worker interrumpido")
            job["ended_at"] = time.time()
            job["elapsed_seconds"] = job["ended_at"] - now
            state["elapsed_wall_seconds"] = time.time() - state["started_at"]
            state["active_seconds"] = sum(j.get("elapsed_seconds", 0) for j in state["jobs"])
            write_json(ledger_path, state)
        return job
