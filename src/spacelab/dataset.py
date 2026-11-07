"""Lazy image loading and explicit, training-only augmentation recipes."""
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

from .io import digest


def verify_record(root, row):
    for key,sha_key in [("image","image_sha256"),("label","label_sha256")]:
        if sha_key in row and digest(Path(root)/row[key]) != row[sha_key]:
            raise ValueError(f"Original modificado desde la partición: {row[key]}")


def read_labels(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        v = np.array([float(x) for x in line.split()])
        if len(v) != 5 or not np.isfinite(v).all():
            raise ValueError(f"Etiqueta inválida: {path}")
        c, x, y, w, h = v
        if c != int(c) or c not in range(5) or min(w, h) <= 0 or min(x-w/2,y-h/2) < -1e-6 or max(x+w/2,y+h/2) > 1+1e-6:
            raise ValueError(f"Caja inválida: {path}: {line}")
        rows.append(v)
    return np.array(rows, dtype=np.float32).reshape(-1, 5)


def letterbox(image, size):
    w, h = image.size
    scale = size / max(w, h)
    nw, nh = max(1, round(w*scale)), max(1, round(h*scale))
    dx, dy = (size-nw)//2, (size-nh)//2
    canvas = Image.new("RGB", (size, size), (114,114,114))
    canvas.paste(image.resize((nw,nh), Image.Resampling.BILINEAR), (dx,dy))
    return canvas, (nw/w, nh/h, dx, dy)


def augment(image, labels, recipe, rng):
    labels = labels.copy()
    if recipe == "geometric":
        k = rng.randrange(4)
        # np.rot90: CCW in array coordinates; each step maps (x,y) -> (y,1-x).
        array = np.rot90(np.asarray(image), k).copy()
        image = Image.fromarray(array)
        for _ in range(k):
            x, y, w, h = [labels[:,j].copy() for j in range(1,5)]
            labels[:,1], labels[:,2], labels[:,3], labels[:,4] = y, 1-x, h, w
        if rng.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            labels[:,1] = 1-labels[:,1]
        if rng.random() < .5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            labels[:,2] = 1-labels[:,2]
    elif recipe == "photometric":
        for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
            image = enhancer(image).enhance(rng.uniform(.85,1.15))
    elif recipe != "baseline":
        raise ValueError(f"Augmentation desconocida: {recipe}")
    return image, labels


class DetectionDataset(Dataset):
    def __init__(self, root, records, config, training=False, epoch=0):
        self.root, self.records, self.config = Path(root), records, config
        self.training, self.epoch = training, epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        t = time.perf_counter()
        row = self.records[index]
        verify_record(self.root,row)
        after_verify = time.perf_counter()
        with Image.open(self.root / row["image"]) as im:
            image = im.convert("RGB")
        read_seconds = time.perf_counter()-after_verify
        labels = read_labels(self.root / row["label"])
        if self.training:
            rng = random.Random(self.config["seed"] + self.epoch*1000003 + index)
            image, labels = augment(image, labels, self.config["augmentation"], rng)
        w, h = image.size
        image, (sx,sy,dx,dy) = letterbox(image, self.config["imgsz"])
        size = self.config["imgsz"]
        labels[:,1] = (labels[:,1]*w*sx+dx)/size
        labels[:,2] = (labels[:,2]*h*sy+dy)/size
        labels[:,3] *= w*sx/size
        labels[:,4] *= h*sy/size
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
        return tensor, torch.from_numpy(labels), {"image_read_seconds":read_seconds,
            "verify_seconds":after_verify-t,"total_prepare_seconds":time.perf_counter()-t}


def collate(batch):
    images, labels, reads = zip(*batch)
    timings = {key:sum(r[key] for r in reads) for key in reads[0]}
    return torch.stack(images), list(labels), timings


def oversample(records, factor=2):
    if factor < 1 or factor > 3 or factor != int(factor):
        raise ValueError("Sobremuestreo entero controlado entre 1 y 3")
    expanded = [r for r in records for _ in range(int(factor) if r["counts"][4] else 1)]
    return expanded
