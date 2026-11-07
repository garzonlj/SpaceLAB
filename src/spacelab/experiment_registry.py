"""Config loading, immutable runs, comparable rankings and final configuration freeze."""
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from .io import digest, fingerprint, read_json, write_json

REPO = Path(__file__).resolve().parents[2]


def config_from(path, selected_run=None):
    base = yaml.safe_load((REPO/"configs/global.yaml").read_text(encoding="utf-8"))["training"]
    variant = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    config = {**base,**variant}
    if config["model"] == "selected":
        if not selected_run:
            raise ValueError("Configure SELECTED_RUN con un candidato completo y comparable de arquitectura")
        manifest = read_json(Path(selected_run)/"experiment_manifest.json")
        if manifest["status"] != "completado" or manifest.get("oom_recovery"):
            raise ValueError("El candidato seleccionado no está completo o cambió de batch")
        if not (Path(selected_run)/"metrics/official_score.json").is_file():
            raise ValueError("Primero validar el candidato seleccionado")
        config["model"] = manifest["config"]["model"]
        config["selection_parent"] = str(Path(selected_run).resolve())
    if config["imgsz"] not in (640,896,1024) or config["seed"] != 42:
        raise ValueError("Resolución/semilla fuera del protocolo declarado")
    return config


def signature(config, protocol, effective_batch=None):
    # Deliberately exclude the variable being tested and include all other recipe
    # fields. Different experiment families form different strict-comparison cohorts.
    c = dict(config)
    for key in ["id","enabled","cap_seconds","priority","hypothesis","selection_parent"]:
        c.pop(key,None)
    c["effective_batch"] = effective_batch or config["batch"]
    variable = c.pop("comparison_variable","model")
    fields = {"model":["model"],"resolution":["imgsz"],"augmentation":["augmentation"],
        "freeze":["freeze","unfreeze_epoch"],"lr":["lr"],"optimizer":["optimizer"],
        "regularization":["weight_decay","patience","augmentation","freeze"],
        "oversampling":["oversample_factor"],"inference":["nms_iou","tile_size","tile_overlap"]}[variable]
    for key in fields:
        c.pop(key,None)
    return fingerprint({"protocol_id":protocol["protocol_id"],"variable":variable,"recipe":c})


def collect(storage):
    rows = []
    for path in Path(storage).glob("jobs/**/experiment_manifest.json"):
        m = read_json(path)
        score_path = path.parent/"metrics/official_score.json"
        row = {"run_id":path.parent.name,"path":str(path.parent),"status":m["status"],
               "model":m["config"]["model"],"id":m["config"]["id"],"phase":m["config"]["phase"],
               "cohort":m["cohort"],"protocol_id":m["protocol_id"],
               "comparable":m["status"]=="completado" and not m.get("oom_recovery"),
               "train_wall_seconds":m.get("train_wall_seconds"),"peak_vram_bytes":m.get("peak_vram_bytes")}
        if score_path.is_file():
            score = read_json(score_path)
            if score.get("evaluation_split","internal_val") != "internal_val":
                continue
            row.update({k:v for k,v in score["total"].items() if k!="por_clase"})
            for name,v in score["total"]["por_clase"].items():
                row[name+"_ap"] = v["ap50_95"]
                row[name+"_f1"] = v["f1"]
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty and "score" in table:
        # Cohort-local ranks only. Never rank mixed epochs/data/batches together.
        eligible = table.comparable & table.score.notna()
        table.loc[eligible,"cohort_rank"] = table.loc[eligible].groupby("cohort")["score"].rank(ascending=False,method="min")
    return table


def freeze_selection(run, protocol, destination, inference_run=None):
    run = Path(run)
    manifest = read_json(run/"experiment_manifest.json")
    if manifest["status"] != "completado" or manifest.get("oom_recovery"):
        raise ValueError("Solo se congela un modelo completo sin cambios de batch")
    if manifest["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Protocolo diferente")
    if Path(destination).exists():
        raise FileExistsError("Selección final ya congelada; no se reemplaza tras consultar test")
    checkpoint = run/"checkpoints/best.pt"
    if inference_run:
        provenance = read_json(Path(inference_run)/"validation_provenance.json")
        if (Path(provenance["training_run"]).resolve() != run.resolve()
            or provenance["protocol_id"] != protocol["protocol_id"]
            or provenance["checkpoint_sha256"] != digest(checkpoint)):
            raise ValueError("La receta de inferencia corresponde a otro modelo/protocolo")
        evaluation = read_json(Path(inference_run)/"metrics/official_score.json")
    else:
        evaluation = read_json(run/"metrics/official_score.json")
    if evaluation.get("evaluation_split") != "internal_val":
        raise ValueError("Solo se congela una receta seleccionada con validación interna")
    recipe = evaluation.get("inference_config",manifest["config"])
    if recipe.get("ensemble_run"):
        extra = Path(recipe["ensemble_run"])
        extra_manifest = read_json(extra/"experiment_manifest.json")
        if (extra_manifest["status"] != "completado" or extra_manifest["protocol_id"] != protocol["protocol_id"]
            or digest(extra/"checkpoints/best.pt") != recipe["ensemble_checkpoint_sha256"]):
            raise ValueError("No se puede congelar un ensemble con checkpoint incompatible.")
    # Freeze hash, checkpoint, threshold and inference configuration BEFORE any test.
    frozen = {"created_at":time.time(),"config":manifest["config"],"protocol_id":protocol["protocol_id"],
        "checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":digest(checkpoint),
        "threshold":evaluation["umbral"],"selected_internal_score":evaluation["total"]["score"],
        "inference_config":evaluation.get("inference_config",manifest["config"])}
    frozen["selection_id"] = fingerprint(frozen)
    write_json(destination,frozen)
    return frozen
