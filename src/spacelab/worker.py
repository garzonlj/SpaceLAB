"""Subprocess entrypoint, invoked only by an explicitly enabled notebook action."""
import gc
import shutil
import sys
import time
import traceback
from pathlib import Path

from filelock import FileLock

from .io import dataset_root, digest, fingerprint, read_json, write_json
from .time_budget import Deadline, DeadlineExceeded


def run(request):
    from .experiment_registry import signature
    from .splits import create_protocol, load_protocol, select_records
    from .training import benchmark, calibration_key, environment, estimate, train
    folder = Path(request["run"])
    p = request["payload"]
    action = request["action"]
    root = dataset_root(p["dataset"])
    deadline = Deadline(request["stop_at"])
    storage = Path(request["storage"])
    protocol_path = storage/"protocol.json"
    if action == "splits":
        protocol = create_protocol(root,protocol_path,deadline,p.get("groups_csv"))
        return {"status":"completado","protocol_id":protocol["protocol_id"],"distributions":protocol["distributions"]}
    protocol = load_protocol(protocol_path,root)
    if action == "benchmark":
        config = p["config"]
        if not config["enabled"] and not p.get("enable_optional",False):
            return {"status":"omitido","reason":"Configuración opcional desactivada"}
        measured = benchmark(config,root,protocol,deadline,folder)
        write_json(storage/"calibration"/(calibration_key(config)+".json"),measured)
        return {"status":"completado","calibration":measured}
    if action == "train":
        config = p["config"]
        if not config["enabled"] and not p.get("enable_optional",False):
            return {"status":"omitido","reason":"Configuración opcional desactivada"}
        calibration_path = storage/"calibration"/(calibration_key(config)+".json")
        if not calibration_path.exists():
            return {"status":"omitido","reason":"Calibración pendiente para modelo/resolución/batch/AMP/augmentation/freeze/optimizador"}
        calibration = read_json(calibration_path)
        if calibration["environment"]["gpu"] != environment()["gpu"]:
            return {"status":"omitido","reason":"La calibración procede de otra GPU"}
        projection = estimate(config,protocol,calibration)
        if projection > deadline.remaining()-20:
            return {"status":"omitido","reason":"La proyección medida no cabe en el techo asignado", "projected_seconds":projection}
        manifest = {"status":"ejecutándose","config":config,"protocol_id":protocol["protocol_id"],
            "cohort":signature(config,protocol),"projected_seconds":projection}
        manifest["source_sha256"] = {path.name:digest(path) for path in Path(__file__).parent.glob("*.py")}
        write_json(folder/"experiment_manifest.json",manifest)
        try:
            if p.get("resume"):
                previous = Path(p["resume"]).parent/"best.pt"
                if previous.exists():
                    (folder/"checkpoints").mkdir(exist_ok=True)
                    shutil.copy2(previous,folder/"checkpoints/best.pt")
            result = train(config,root,protocol,deadline,folder,p.get("resume"))
            manifest.update(result)
            manifest["cohort"] = signature(config,protocol,result["effective_batch"])
            write_json(folder/"experiment_manifest.json",manifest)
            return {**result,"run":str(folder)}
        except BaseException as error:
            manifest.update(status="interrumpido",reason=str(error))
            write_json(folder/"experiment_manifest.json",manifest)
            raise
    if action in ("validate","inference","test","submission"):
        return evaluate_or_submit(request,root,protocol,deadline)
    if action == "threshold":
        from .evaluation import official_modules,score_predictions,save_evaluation
        source = Path(p["validation_run"])
        provenance = read_json(source/"validation_provenance.json")
        if provenance["protocol_id"] != protocol["protocol_id"]:
            raise ValueError("Protocolo diferente")
        ev,_ = official_modules()
        deadline.check(15)
        predictions_source = provenance.get("predictions_source",str(source/"predictions"))
        preds = ev.leer_envio(predictions_source).preds
        records = select_records(protocol,"val")
        inference_config = dict(provenance["inference_config"])
        if p.get("threshold_grid") is not None:
            inference_config["threshold_grid"] = p["threshold_grid"]
        result,gt,pc = score_predictions(root,records,preds,tune=True,thresholds=inference_config.get("threshold_grid"))
        result.update(evaluation_split="internal_val",inference_config=inference_config)
        save_evaluation(folder,root,records,preds,result,gt,pc,{"inference_seconds":0,"cached_predictions":True})
        write_json(folder/"validation_provenance.json",{**provenance,"inference_config":inference_config,"predictions_source":predictions_source})
        return {"status":"completado","metrics":result,"run":str(folder)}
    if action == "cross_validation":
        folds = protocol.get("grouped_folds")
        if not protocol["groups_verified"] or not folds:
            return {"status":"omitido","reason":"Dos folds requieren grupos verificados"}
        if not p.get("enable_optional"):
            return {"status":"omitido","reason":"CV desactivada"}
        config = p["config"]
        results = []
        calibration_path = storage/"calibration"/(calibration_key(config)+".json")
        if not calibration_path.exists():
            return {"status":"omitido","reason":"Calibración pendiente"}
        for index,(tr,va) in enumerate(folds):
            deadline.check(30)
            sub = {**protocol,"train_ids":tr,"val_ids":va,"screening_ids":tr,
                   "protocol_id":protocol["protocol_id"]+f"-fold-{index}"}
            fold_folder = folder/f"fold_{index}"
            fold_folder.mkdir(exist_ok=False)
            fold_config = {**config,"phase":"finalist"}
            if estimate(fold_config,sub,read_json(calibration_path)) > deadline.remaining()-30:
                return {"status":"interrumpido" if results else "omitido","reason":"Fold no cabe en presupuesto","folds":results}
            manifest = {"config":fold_config,"status":"ejecutándose","protocol_id":sub["protocol_id"],
                "parent_protocol_id":protocol["protocol_id"],"protocol_snapshot":sub,
                "cohort":signature(fold_config,sub)}
            write_json(fold_folder/"experiment_manifest.json",manifest)
            result = train(fold_config,root,sub,deadline,fold_folder)
            manifest.update(result)
            write_json(fold_folder/"experiment_manifest.json",manifest)
            results.append(result)
            if result["status"] != "completado":
                return {"status":"interrumpido","folds":results}
        write_json(folder/"two_fold_results.json",results)
        return {"status":"completado","folds":results}
    raise ValueError(f"Acción desconocida: {action}")


def evaluate_or_submit(request,root,protocol,deadline):
    import torch
    from .checkpointing import restore
    from .evaluation import official_modules, score_predictions, save_evaluation
    from .inference import predictions
    from .models import Detector, DetectorEnsemble
    from .splits import select_records
    from .training import environment
    folder = Path(request["run"])
    p,action = request["payload"],request["action"]
    final = action in ("test","submission")
    if final:
        frozen = read_json(Path(request["storage"])/"frozen_selection.json")
        if fingerprint({k:v for k,v in frozen.items() if k!="selection_id"}) != frozen["selection_id"]:
            raise ValueError("Contenido de selección congelada alterado")
        if frozen["protocol_id"] != protocol["protocol_id"] or digest(frozen["checkpoint"]) != frozen["checkpoint_sha256"]:
            raise ValueError("Selección congelada alterada")
        checkpoint,config = frozen["checkpoint"],frozen["config"]
        inference_config = frozen["inference_config"]
        threshold = frozen["threshold"]
        marker = Path(request["storage"])/("official_test_started.json" if action=="test" else "submission_started.json")
        if marker.exists():
            raise ValueError("Acción final ya iniciada. No repetir para adaptar la selección al test.")
        write_json(marker,{"selection_id":frozen["selection_id"],"started_at":time.time(),"run":str(folder)})
        split = "test" if action=="test" else "oculto"
        records = [{"id":f.stem,"image":f.relative_to(root).as_posix(),
                    "label":f"labels/test/{f.stem}.txt"} for f in sorted((root/"images"/split).glob("*.jpg"))]
        if action=="test":
            from PIL import Image
            for row in records:
                deadline.check(5)
                with Image.open(root/row["image"]) as image:
                    row.update(width=image.width,height=image.height)
    else:
        source = Path(p["run"])
        manifest = read_json(source/"experiment_manifest.json")
        config = manifest["config"]
        if manifest["protocol_id"] != protocol["protocol_id"]:
            if manifest.get("parent_protocol_id") != protocol["protocol_id"]:
                raise ValueError("Protocolo distinto")
            protocol = manifest["protocol_snapshot"]
        checkpoint = str(source/"checkpoints/best.pt")
        inference_config = {**config,**p.get("inference_overrides",{})}
        allowed = {"nms_iou","tile_size","tile_overlap","tile_merge_iou","confidence_floor",
                   "yolo_head","tta","post_nms_iou","inference_imgsz","threshold_grid",
                   "ensemble_run","ensemble_checkpoint_sha256"}
        if set(p.get("inference_overrides",{}))-allowed:
            raise ValueError("La ablación de inferencia no puede modificar modelo, datos o receta de entrenamiento")
        threshold = None
        records = select_records(protocol,"val")
        if p.get("probe_ids") is not None:
            probe_ids = set(p["probe_ids"])
            if not probe_ids or not probe_ids <= {r["id"] for r in records}:
                raise ValueError("Las pruebas de inferencia solo pueden usar validación interna.")
            records = [r for r in records if r["id"] in probe_ids]
    device = p.get("device","cuda")
    if device not in ("cuda","cpu"):
        raise ValueError("Dispositivo de inferencia desconocido")
    print("Dispositivo de inferencia:",device,flush=True)
    detector = Detector(config,pretrained=False).to(device)
    ensemble_detector = None
    try:
        restore(checkpoint,detector,config,protocol["protocol_id"])
        detector.config = inference_config
        if detector.end2end and inference_config.get("yolo_head","native") != "one2many" and inference_config["nms_iou"] != config["nms_iou"]:
            raise ValueError("NMS IoU no es operativo para salida end-to-end; no comparar como cambio efectivo")
        if inference_config.get("ensemble_run"):
            extra_run = Path(inference_config["ensemble_run"])
            extra_manifest = read_json(extra_run/"experiment_manifest.json")
            extra_checkpoint = extra_run/"checkpoints/best.pt"
            if (extra_manifest["protocol_id"] != protocol["protocol_id"]
                or extra_manifest["status"] != "completado"
                or digest(extra_checkpoint) != inference_config["ensemble_checkpoint_sha256"]):
                raise ValueError("Checkpoint del ensemble incompatible o modificado")
            ensemble_detector = Detector(extra_manifest["config"],pretrained=False).to(device)
            restore(extra_checkpoint,ensemble_detector,extra_manifest["config"],protocol["protocol_id"])
            ensemble_detector.config = inference_config
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        if not final:
            # Independently regenerate actual history plots alongside this evaluation.
            history = Path(p["run"])/"metrics/training_history.csv"
            if history.exists():
                (folder/"metrics").mkdir(exist_ok=True)
                shutil.copy2(history,folder/"metrics/training_history.csv")
        predictor = DetectorEnsemble([detector,ensemble_detector]) if ensemble_detector is not None else detector
        preds,infer_time = predictions(predictor,root,records,inference_config,deadline,folder/"predictions")
        deadline.check(10)
        if action=="submission":
            _,fe = official_modules()
            fe.crear_zip(str(folder/"predictions"),str(folder/"submission.zip"),umbral=threshold)
            if not fe.verificar(str(folder/"submission.zip"),str(root/"images/oculto")):
                raise ValueError("ZIP rechazado por el verificador oficial")
            return {"status":"completado","zip":str(folder/"submission.zip")}
        result,gt,pc = score_predictions(root,records,preds,threshold=threshold,tune=not final,
                                       thresholds=inference_config.get("threshold_grid"))
        result["evaluation_split"] = "official_test" if final else ("internal_val_probe" if p.get("probe_ids") else "internal_val")
        result["inference_config"] = inference_config
        train_runtime_path = Path(p["run"])/"metrics/training_runtime.json" if not final else None
        runtime = {"inference_seconds":infer_time,"peak_vram_bytes":torch.cuda.max_memory_allocated() if device=="cuda" else 0,
            "environment":environment() if device=="cuda" else {"device":"cpu","torch":torch.__version__},"training_runtime":read_json(train_runtime_path)
            if train_runtime_path and train_runtime_path.exists() else None}
        if p.get("fast_metrics_only") and not final:
            write_json(folder/"metrics/official_score.json",result)
            write_json(folder/"metrics/runtime.json",runtime)
        else:
            save_evaluation(folder,root,records,preds,result,gt,pc,runtime)
        if not final:
            # Each validation remains immutable; source gets only a pointer and
            # score for registry selection, never overwritten prediction artifacts.
            if action=="validate" and not p.get("probe_ids"):
                target = Path(p["run"])/"metrics/official_score.json"
                if not target.exists():
                    write_json(target,result)
            write_json(folder/"validation_provenance.json",{"training_run":p["run"],"checkpoint_sha256":digest(checkpoint),
                "protocol_id":protocol["protocol_id"],"inference_config":inference_config,
                "validation_ids":[r["id"] for r in records]})
        return {"status":"completado","run":str(folder),"metrics":result}
    finally:
        del detector,ensemble_detector
        gc.collect()
        torch.cuda.empty_cache()


def main():
    request = read_json(sys.argv[1])
    folder = Path(request["run"])
    try:
        # Independent GPU lock prevents an orphan worker from sharing the T4
        # with another notebook if its original supervisor kernel disappeared.
        lock_name = "cpu_delivery.lock" if request["payload"].get("device") == "cpu" else "gpu.lock"
        with FileLock(str(Path(request["storage"])/lock_name),timeout=0):
            Deadline(request["stop_at"]).check()
            result = run(request)
        write_json(folder/"result.json",result)
    except BaseException as error:
        write_json(folder/"result.json",{"status":"interrumpido","reason":str(error),"traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
