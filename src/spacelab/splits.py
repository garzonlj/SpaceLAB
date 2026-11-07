"""One immutable development protocol; never use official test for selection."""
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .io import CLASSES, digest, fingerprint, read_json, write_json


def balanced_partition(records, fraction, seed, group_key="id"):
    """Greedy repeated holdout, balanced on presence, counts, area bins and sources.

    A heuristic, not exact stratification; persist achieved distributions and residuals.
    Units are indivisible when reliable groups are supplied.
    """
    rng = random.Random(seed)
    sources = sorted({r["source"] for r in records})
    units = defaultdict(list)
    for r in records:
        units[r[group_key]].append(r)
    keys = sorted(units)
    vectors = {}
    for k, rows in units.items():
        vectors[k] = np.array([len(rows),
            *np.sum([np.array(r["counts"]) > 0 for r in rows], axis=0),
            *np.sum([r["counts"] for r in rows], axis=0),
            *np.sum([r["area_bins"] for r in rows], axis=0),
            *[sum(r["source"] == source for r in rows) for source in sources]], dtype=float)
    total = sum(vectors.values())
    target, denominator = total*fraction, np.maximum(total*fraction, 1)
    best = None
    for _ in range(48):
        rng.shuffle(keys)
        chosen, current = set(), np.zeros_like(total)
        for k in keys:
            candidate = current + vectors[k]
            if np.mean(((candidate-target)/denominator)**2) < np.mean(((current-target)/denominator)**2):
                chosen.add(k)
                current = candidate
        cost = np.mean(((current-target)/denominator)**2)
        if chosen and len(chosen) < len(keys) and (best is None or cost < best[0]):
            best = (cost, chosen.copy())
    if best is None:
        raise ValueError("No es posible dividir unidades sin romper grupos")
    right = [r for r in records if r[group_key] in best[1]]
    left = [r for r in records if r[group_key] not in best[1]]
    for subset in (left,right):
        if (np.sum([r["counts"] for r in subset],axis=0) == 0).any():
            raise ValueError("Una clase desaparece del split; revise los grupos/fracción")
    return left, right


def describe(records):
    return {"images": len(records), "objects_by_class": np.sum([r["counts"] for r in records],axis=0).tolist(),
            "images_by_class": np.sum([np.array(r["counts"])>0 for r in records],axis=0).tolist(),
            "area_bins": np.sum([r["area_bins"] for r in records],axis=0).tolist()}


def create_protocol(root, destination, deadline, groups_csv=None, screening_fraction=.25, seed=42):
    from .dataset import read_labels

    root, destination = Path(root), Path(destination)
    if destination.exists():
        return load_protocol(destination, root)  # Never silently regenerate.
    reliable = {}
    if groups_csv:
        with Path(groups_csv).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("verified", "").lower() != "true" or not row.get("group"):
                    raise ValueError("Cada grupo necesita verified=true e id,group no vacíos")
                if row["id"] in reliable:
                    raise ValueError("ID repetido en grupos")
                reliable[row["id"]] = row["group"]
    records = []
    candidates = {}
    seen_hashes = {}
    for path in sorted((root/"images/train").glob("*.jpg")):
        deadline.check(15)
        label = root/"labels/train"/(path.stem+".txt")
        y = read_labels(label)
        with Image.open(path) as image:
            w,h = image.size
        area = y[:,3]*w*y[:,4]*h
        sha = digest(path)
        if sha in seen_hashes:
            raise ValueError("Duplicado exacto detectado: aportar grupos verificados antes de dividir")
        seen_hashes[sha] = path.stem
        candidate = re.sub(r"__t\d+_\d+$", "", path.stem)
        candidates[path.stem] = candidate
        records.append({"id": path.stem, "image": path.relative_to(root).as_posix(),
            "label": label.relative_to(root).as_posix(), "source": path.stem.split("__")[0],
            "group": reliable.get(path.stem, path.stem), "width": w, "height": h,
            "counts": np.bincount(y[:,0].astype(int),minlength=5).tolist(),
            "area_bins": [int((area < 32**2).sum()), int(((area>=32**2)&(area<96**2)).sum()), int((area>=96**2).sum())],
            "image_sha256": sha, "label_sha256": digest(label)})
    if reliable and set(reliable) != {r["id"] for r in records}:
        raise ValueError("El CSV verificado debe cubrir exactamente todo train")
    if len(records) != 5926:
        raise ValueError(f"Entrega train incompleta o distinta del EDA: {len(records)} imágenes, se esperaban 5926")
    train, val = balanced_partition(records,.2,seed,"group" if reliable else "id")
    # Screening is fixed, disjoint from validation, and respects verified groups.
    _, screen = balanced_partition(train,screening_fraction,seed+1,"group" if reliable else "id")
    folds = None
    if reliable:
        a,b = balanced_partition(train,.5,seed+2,"group")
        folds = [[ [r["id"] for r in a], [r["id"] for r in b] ],
                 [ [r["id"] for r in b], [r["id"] for r in a] ]]
    tr_ids, va_ids = {r["id"] for r in train}, {r["id"] for r in val}
    overlap_candidates = sorted(set(candidates[i] for i in tr_ids) & set(candidates[i] for i in va_ids))
    protocol = {"version":1,"seed":seed,"classes":CLASSES,"records":records,
        "train_ids":sorted(tr_ids),"val_ids":sorted(va_ids),"screening_ids":sorted(r["id"] for r in screen),
        "groups_verified":bool(reliable),"grouped_folds":folds,"candidate_scene_overlap":overlap_candidates,
        "distributions":{key:describe(value) for key,value in [("train",train),("val",val),("screening",screen)]},
        "limitations":"Sin grupos verificados: estratificación multilabel heurística; independencia de escenas no demostrada.",
        "official_test_reserved":True,"yaml_sha256":digest(root/"data.yaml")}
    protocol["protocol_id"] = fingerprint(protocol)
    assert not tr_ids & va_ids
    if reliable:
        assert not {r["group"] for r in train} & {r["group"] for r in val}
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Atomic create; concurrent creation is additionally prevented by the budget/GPU lock.
    write_json(destination,protocol)
    return protocol


def load_protocol(path, root=None):
    protocol = read_json(path)
    actual = fingerprint({k:v for k,v in protocol.items() if k != "protocol_id"})
    if actual != protocol["protocol_id"] or protocol["classes"] != CLASSES:
        raise ValueError("Protocolo alterado o clases incompatibles")
    if set(protocol["train_ids"]) & set(protocol["val_ids"]):
        raise ValueError("Solapamiento de IDs")
    if root and digest(Path(root)/"data.yaml") != protocol["yaml_sha256"]:
        raise ValueError("El data.yaml original ha cambiado")
    return protocol


def select_records(protocol, subset):
    ids = set(protocol[subset+"_ids"])
    return [r for r in protocol["records"] if r["id"] in ids]
