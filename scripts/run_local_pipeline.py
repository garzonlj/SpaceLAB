"""Run the experimental matrix sequentially on the local CUDA GPU.

Local runs have no 90-minute session limit. Each worker still has a watchdog.
Resume by passing the same --storage; completed actions are not repeated.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO / ".eda_tmp/ultralytics"))
os.environ.setdefault("TORCH_HOME", str(REPO / "artifacts/torch"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".eda_tmp/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main():
    import torch
    import yaml
    from spacelab.experiment_registry import config_from, collect, freeze_selection
    from spacelab.io import read_json, write_json
    from spacelab.preparation import verify_layout
    from spacelab.splits import load_protocol
    from spacelab.time_budget import launch
    from spacelab.training import calibration_key, environment, estimate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(REPO / "spacelab_dataset"))
    parser.add_argument("--storage", default=str(REPO / "reports/experiments" / ("local_" + datetime.now().strftime("%Y%m%d_%H%M%S"))))
    parser.add_argument("--deadline", help="Absolute ISO datetime including UTC offset.")
    parser.add_argument("--quick", action="store_true", help="Prioritize architecture comparison and final delivery.")
    parser.add_argument("--focus-nano", action="store_true", help="Continue the trained nano on all internal training data.")
    parser.add_argument("--adopt-run", help="Wait for the existing worker after replacing its supervisor.")
    args = parser.parse_args()
    root = verify_layout(args.dataset)
    storage = Path(args.storage).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    env = environment()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    write_json(storage / "environment.json", env)
    print("Dataset:", root, "\nResultados:", storage, "\nGPU:", env["gpu"], flush=True)
    print("Sesión local; batch=1, workers=0, AMP=False (FP32).", flush=True)
    state_path = storage / "pipeline_state.json"
    state = read_json(state_path) if state_path.exists() else {"dataset": str(root), "jobs": {}, "omissions": []}
    stop_at = datetime.fromisoformat(args.deadline).timestamp() if args.deadline else None
    if stop_at is not None:
        write_json(storage / "runtime_deadline.json", {"stop_at": stop_at, "deadline": args.deadline,
                   "final_reserve_seconds": 900, "quick": args.quick})
        print("PLAZO MÁXIMO:", args.deadline, "· reserva de entrega: 15 minutos", flush=True)
    if args.adopt_run:
        from filelock import FileLock
        adopted = Path(args.adopt_run).resolve()
        request = read_json(adopted / "request.json")
        if Path(request["storage"]).resolve() != storage or request["action"] not in ("train", "validate"):
            raise ValueError("Solo se adopta un entrenamiento o validación de esta sesión.")
        print("Conservando trabajo en curso:", adopted, flush=True)
        while not (adopted / "result.json").exists():
            if stop_at is not None and time.time() >= stop_at - 900:
                raise RuntimeError("El entrenamiento adoptado no terminó antes de la reserva final.")
            time.sleep(5)
        result = read_json(adopted / "result.json")
        with FileLock(str(storage / "budget.lock"), timeout=30):
            ledger = read_json(storage / "budget.json")
            adopted_job = next(j for j in ledger["jobs"] if Path(j.get("run", "")).resolve() == adopted)
            adopted_job.update(status=result["status"], result=result, ended_at=time.time(),
                               elapsed_seconds=time.time()-adopted_job["started_at"], returncode=0 if result["status"]=="completado" else 1)
            write_json(storage / "budget.json", ledger)
        if request["action"] == "train":
            adopted_key = request["payload"]["config"]["id"]
        else:
            adopted_manifest = read_json(Path(request["payload"]["run"]) / "experiment_manifest.json")
            adopted_key = adopted_manifest["config"]["id"] + "_validate"
        state["jobs"][adopted_key] = adopted_job
        write_json(state_path, state)
        print("Entrenamiento conservado:", adopted_job["status"], flush=True)

    def available(reserve=900):
        return float("inf") if stop_at is None else max(0, stop_at-time.time()-reserve)

    def job(key, action, payload, cap=86400, final=False):
        previous = state["jobs"].get(key)
        if previous and previous["status"] == "completado":
            request_path = Path(previous["run"]) / "request.json"
            previous_request = read_json(request_path) if request_path.exists() else {}
            if previous_request.get("payload") == {"dataset": str(root), **payload}:
                print(key, "ya completado:", previous.get("run"), flush=True)
                return previous
        print("INICIO", key, action, flush=True)
        write_json(storage / "current_action.json", {"key": key, "action": action, "started_at": datetime.now().isoformat()})
        result = launch(action, {"dataset": str(root), **payload}, storage, cap_seconds=cap, final=final, unlimited=True)
        state["jobs"][key] = result
        write_json(state_path, state)
        print("FIN", key, result["status"], result.get("reason") or "", flush=True)
        if result.get("run") and result["status"] != "completado":
            log = Path(result["run"]) / "worker.log"
            if log.exists():
                print(log.read_text(encoding="utf-8", errors="replace")[-4000:], flush=True)
        return result

    def omission(key, reason):
        print("OMITIDO", key, reason, flush=True)
        state["omissions"].append({"id": key, "reason": reason})
        write_json(state_path, state)

    def training(entry, parent=None):
        cfg = config_from(REPO / entry["config"], parent)
        cfg.update(batch=1, workers=0, amp=False, enabled=True)
        key = cfg["id"]
        calibration = storage / "calibration" / (calibration_key(cfg) + ".json")
        if not calibration.exists():
            b = job("calibration_" + calibration_key(cfg), "benchmark", {"config": cfg, "enable_optional": True}, cap=1800)
            if b["status"] != "completado":
                omission(key, "Calibración fallida; revisar worker.log.")
                return None
        projected = estimate(cfg, protocol, read_json(calibration))
        if args.quick and projected + 240 > available():
            omission(key, "La proyección más validación no cabe antes de la reserva de entrega.")
            return None
        result = job(key, "train", {"config": cfg, "enable_optional": True}, cap=max(86400, projected * 3))
        if result["status"] != "completado":
            return None
        validation = job(key + "_validate", "validate", {"run": result["run"]})
        if validation["status"] != "completado":
            return None
        state["omissions"] = [x for x in state["omissions"] if x["id"] != key]
        write_json(state_path, state)
        return result["run"]

    if not (storage / "protocol.json").exists():
        if job("splits", "splits", {}, cap=1800)["status"] != "completado":
            raise RuntimeError("No se creó el protocolo; consulte worker.log.")
    protocol = load_protocol(storage / "protocol.json", root)
    catalog = yaml.safe_load((REPO / "configs/experiments.yaml").read_text(encoding="utf-8"))["experiments"]
    by_id = {e["id"]: e for e in catalog}
    completed = []
    architecture_ids = (("M01",) if args.focus_nano else
                        (("M01", "O01", "M02") if args.quick else ("M01", "M02")))
    for key in architecture_ids:
        if args.quick and key == "M02" and available() < 2400:
            omission(key, "Se prioriza YOLO26s por indicación del equipo y se reserva tiempo para el finalista.")
            continue
        candidate = training(by_id[key])
        if candidate:
            completed.append(candidate)
    if not completed:
        raise RuntimeError("Ninguna arquitectura completó entrenamiento y validación.")

    def score(run):
        return read_json(Path(run) / "metrics/official_score.json")["total"]["score"]

    selected = max(completed, key=score)
    write_json(storage / "architecture_selection.json", {"run": selected, "score": score(selected)})
    print("Arquitectura seleccionada:", selected, flush=True)
    for entry in catalog:
        key = entry["id"]
        if key.startswith(("M", "I", "C")) or key == "FIN" or key in architecture_ids:
            continue
        if args.quick:
            omission(key, "Se prioriza comparar YOLO26n/YOLO26s y generar la entrega dentro del plazo.")
            continue
        if args.quick and available() < 2400:
            omission(key, "Se prioriza el entrenamiento final y la entrega dentro del plazo.")
            continue
        try:
            candidate = training(entry, selected)
            if candidate:
                completed.append(candidate)
        except Exception as error:
            omission(key, str(error))
    if not args.quick and protocol.get("groups_verified") and protocol.get("grouped_folds"):
        cfg = config_from(REPO / by_id["C01"]["config"], selected)
        cfg.update(batch=1, workers=0, amp=False, enabled=True)
        calibration = storage / "calibration" / (calibration_key(cfg) + ".json")
        if not calibration.exists():
            job("cv_calibration", "benchmark", {"config": cfg, "enable_optional": True}, cap=1800)
        job("C01", "cross_validation", {"config": cfg, "enable_optional": True})
    else:
        omission("C01", "No hay grupos verificados; no se inventan grupos para validación cruzada.")
    # Select hyperparameters on internal validation only, then refit the finalist.
    best_screening = max(completed, key=score)
    finalist_cfg = config_from(REPO / by_id["FIN"]["config"], selected)
    best_cfg = read_json(Path(best_screening) / "experiment_manifest.json")["config"]
    finalist_cfg.update({k: v for k, v in best_cfg.items() if k not in ("id", "phase", "epochs", "comparison_variable", "hypothesis", "selection_parent")})
    finalist_cfg.update(id="FIN", phase="finalist", epochs=6, batch=1, workers=0, amp=False, enabled=True)
    if args.quick:
        finalist_cfg["epochs"] = 3
    if args.focus_nano:
        from spacelab.io import digest
        initial_checkpoint = Path(best_screening) / "checkpoints/best.pt"
        finalist_cfg.update(epochs=12, initial_checkpoint=str(initial_checkpoint.resolve()),
                            initial_checkpoint_sha256=digest(initial_checkpoint),
                            loss_schedule="native_epoch")
    calibration = storage / "calibration" / (calibration_key(finalist_cfg) + ".json")
    if not calibration.exists():
        job("finalist_calibration", "benchmark", {"config": finalist_cfg, "enable_optional": True}, cap=1800)
    can_refit = calibration.exists()
    if args.quick and can_refit:
        while finalist_cfg["epochs"] > 1 and estimate(finalist_cfg,protocol,read_json(calibration)) + 240 > available():
            finalist_cfg["epochs"] -= 1
        can_refit = estimate(finalist_cfg,protocol,read_json(calibration)) + 240 <= available()
    if can_refit:
        fin = job("FIN", "train", {"config": finalist_cfg, "enable_optional": True})
    else:
        omission("FIN", "No cabe una época completa más validación; se conserva el mejor candidato validado.")
        fin = {"status": "omitido"}
    final_run = best_screening
    if fin["status"] == "completado":
        val = job("FIN_validate", "validate", {"run": fin["run"]})
        if val["status"] == "completado" and score(fin["run"]) >= score(best_screening):
            final_run = fin["run"]
    base_val = next(v["run"] for k, v in state["jobs"].items() if k.endswith("_validate") and v["status"] == "completado" and read_json(Path(v["run"]) / "validation_provenance.json")["training_run"] == final_run)
    inference_results = [base_val]
    if args.focus_nano:
        omission("I01", "El umbral ya se optimizó con la validación interna del candidato.")
        threshold = {"status": "omitido"}
    else:
        threshold = job("I01", "threshold", {"validation_run": base_val})
    if threshold["status"] == "completado":
        inference_results.append(threshold["run"])
    model_name = read_json(Path(final_run) / "experiment_manifest.json")["config"]["model"]
    if args.quick:
        omission("I02", "Se conserva la receta validada para priorizar la entrega en plazo.")
    elif model_name.startswith("yolo26"):
        omission("I02", "El detector end-to-end seleccionado no aplica NMS convencional.")
    else:
        for iou in (0.4, 0.7):
            result = job("I02_" + str(iou), "inference", {"run": final_run, "inference_overrides": {"nms_iou": iou}})
            if result["status"] == "completado":
                inference_results.append(result["run"])
    if args.quick:
        omission("I03", "Inferencia por recortes omitida por el nuevo plazo.")
        tiled = {"status": "omitido"}
    else:
        tiled = job("I03", "inference", {"run": final_run, "inference_overrides": {"tile_size": 1024, "tile_overlap": 0.25}})
    if tiled["status"] == "completado":
        inference_results.append(tiled["run"])
    best_inference = max(inference_results, key=score)
    if not (storage / "frozen_selection.json").exists():
        freeze_selection(final_run, protocol, storage / "frozen_selection.json", best_inference)
    test_result = job("official_test", "test", {}, final=True)
    submission_result = job("submission", "submission", {}, final=True)
    table = collect(storage)
    table.to_csv(storage / "comparison.csv", index=False)
    summary = {"finished_at": datetime.now().isoformat(), "selected_run": final_run,
               "best_inference": best_inference, "omissions": state["omissions"],
               "deadline": args.deadline, "quick_plan": args.quick,
               "test_status": test_result["status"], "submission_status": submission_result["status"]}
    if test_result["status"] != "completado" or submission_result["status"] != "completado":
        write_json(storage / "pipeline_incomplete.json", summary)
        raise RuntimeError("La evaluación final o la entrega no se completaron; revisar sus logs.")
    write_json(storage / "pipeline_completed.json", summary)
    print("PIPELINE TERMINADO. Resultados:", storage, flush=True)


if __name__ == "__main__":
    main()
