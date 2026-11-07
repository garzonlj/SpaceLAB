"""Bounded inference search: fixed internal probe, then full-validation confirmation."""
import random
import time
from pathlib import Path

from .evaluation import official_modules, score_predictions
from .io import digest, read_json, write_json
from .splits import select_records


def refine(storage, root, protocol, training_run, validation_run, job, stop_at, ensemble_run=None):
    storage,root = Path(storage),Path(root)
    threshold_grid = [round(i/100,2) for i in range(91)]
    retuned = job("refine_threshold", "threshold", {"validation_run":validation_run,
                  "threshold_grid":threshold_grid}, cap=240)
    baseline = retuned["run"] if retuned["status"] == "completado" else validation_run
    baseline_score = read_json(Path(baseline)/"metrics/official_score.json")["total"]["score"]
    records = select_records(protocol,"val")
    rng = random.Random(42)
    probe = []
    # Include each class, then fill a reproducible random sample. Never use test.
    for cls in range(5):
        for row in records:
            if row["counts"][cls] and row not in probe:
                probe.append(row)
                break
    remaining = [r for r in records if r not in probe]
    probe.extend(rng.sample(remaining,min(128-len(probe),len(remaining))))
    probe_ids = [r["id"] for r in probe]
    write_json(storage/"inference_probe_ids.json",{"protocol_id":protocol["protocol_id"],"ids":probe_ids,
               "selection":"fixed seed 42, all classes represented; full confirmation required"})
    ev,_ = official_modules()
    provenance = read_json(Path(baseline)/"validation_provenance.json")
    predictions_path = provenance.get("predictions_source",str(Path(baseline)/"predictions"))
    all_preds = ev.leer_envio(predictions_path).preds
    probe_preds = {r["id"]:all_preds[r["id"]] for r in probe}
    initial,_,_ = score_predictions(root,probe,probe_preds,thresholds=threshold_grid)
    rows = [{"id":"baseline","probe_score":initial["total"]["score"],"overrides":{},"full_score":baseline_score}]
    common = {"threshold_grid":threshold_grid}
    candidates = [
        ("one2one",{"yolo_head":"one2one"}),
        ("one2many_nms045",{"yolo_head":"one2many","nms_iou":0.45}),
        ("one2many_nms065",{"yolo_head":"one2many","nms_iou":0.65}),
        ("lower_confidence_floor",{"yolo_head":"one2many","nms_iou":0.6,"confidence_floor":0.0001}),
        ("e2e_post_nms",{"post_nms_iou":0.5}),
        ("tta_hflip",{"yolo_head":"one2many","nms_iou":0.6,"tta":True,"post_nms_iou":0.5}),
        ("resolution896",{"yolo_head":"one2many","nms_iou":0.6,"inference_imgsz":896}),
    ]
    if ensemble_run and Path(ensemble_run).resolve() != Path(training_run).resolve():
        candidates.append(("ensemble_checkpoints",{"yolo_head":"one2many","nms_iou":0.6,
            "ensemble_run":str(Path(ensemble_run).resolve()),
            "ensemble_checkpoint_sha256":digest(Path(ensemble_run)/"checkpoints/best.pt"),"post_nms_iou":0.5}))

    def remaining_time():
        return float("inf") if stop_at is None else stop_at-time.time()

    for name,recipe in candidates:
        # Leave time for full confirmation and the final test/submission.
        if remaining_time() < 1200:
            rows.append({"id":name,"status":"omitido","reason":"Reserva para confirmar y entregar"})
            continue
        result = job("probe_"+name,"inference",{"run":training_run,
                     "inference_overrides":{**common,**recipe},"probe_ids":probe_ids,"fast_metrics_only":True},cap=180)
        row = {"id":name,"overrides":{**common,**recipe},"status":result["status"]}
        if result["status"] == "completado":
            score = read_json(Path(result["run"])/"metrics/official_score.json")
            row.update(probe_score=score["total"]["score"],probe_run=result["run"])
        rows.append(row)
        write_json(storage/"inference_search.json",rows)
    best_run,best_score = baseline,baseline_score
    eligible = sorted([r for r in rows if r.get("overrides") and r.get("probe_score",-1)>initial["total"]["score"]],
                      key=lambda r:r["probe_score"],reverse=True)
    for winner in eligible[:2]:
        if remaining_time() < 1080:
            break
        result = job("confirm_"+winner["id"],"inference",{"run":training_run,
                     "inference_overrides":winner["overrides"]},cap=420)
        if result["status"] == "completado":
            score = read_json(Path(result["run"])/"metrics/official_score.json")["total"]["score"]
            winner.update(full_score=score,confirmation_run=result["run"])
            if score>best_score:
                best_run,best_score = result["run"],score
    write_json(storage/"inference_search.json",rows)
    write_json(storage/"inference_search_selection.json",{"validation_run":best_run,"score":best_score,
               "baseline_score":baseline_score,"confirmed_on_full_validation":True,
               "test_used_for_selection":False})
    print("Inferencia seleccionada por validación completa:",best_run,"score=",best_score,flush=True)
    return best_run
