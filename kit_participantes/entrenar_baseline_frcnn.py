#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entrenar_baseline_frcnn.py — Segundo baseline: Faster R-CNN (torchvision) · SpaceLAB Hackatón FAC+PUJ
=====================================================================================================

Un detector clásico de dos etapas, en PyTorch puro (sin Ultralytics), con pesos preentrenados en COCO.
Hace el mismo ciclo que el baseline de YOLO:
  1. Entrena sobre images/train leyendo directamente las etiquetas en formato YOLO.
  2. Predice images/test y calcula el PUNTAJE OFICIAL local con evaluar.py; elige el umbral que maximiza F1.
  3. Predice images/oculto y genera el ZIP de envío (con formato_envio.py).

Los baselines son solo puntos de partida para verificar que su flujo funciona de punta a punta.
No son una recomendación: pueden usar cualquier modelo o framework.

Necesita: torch y torchvision (ya vienen en Colab y Kaggle), pillow y numpy. evaluar.py y formato_envio.py
deben estar en la misma carpeta.

Ejemplos:
  # Colab/Kaggle T4, ~1.5 h de entrenamiento como máximo
  python entrenar_baseline_frcnn.py --datos /content/spacelab_dataset --horas 1.5

  # Prueba rápida (5 % de los datos, 1 época, 640 px)
  python entrenar_baseline_frcnn.py --datos ./spacelab_dataset --fraccion 0.05 --epocas 1 --imgsz 640

  # Solo generar el envío con unos pesos ya entrenados
  python entrenar_baseline_frcnn.py --datos ./spacelab_dataset --pesos runs_spacelab/frcnn/weights/last.pt

Memoria: algunas imágenes tienen cientos de barcos y disparan el consumo de memoria al entrenar.
Si aparece `CUDA out of memory`, use un --batch pequeño.
"""
from __future__ import annotations

import argparse
import contextlib
import math
import random
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import evaluar as EV  # noqa: E402
import formato_envio as FE  # noqa: E402

import torch  # noqa: E402
from PIL import Image  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NC = len(EV.CLASES)


# --------------------------------------------------------------------------- datos
class DatasetYOLO(torch.utils.data.Dataset):
    """Imágenes + etiquetas YOLO (clase xc yc w h normalizados) -> formato de torchvision (cajas x1y1x2y2 en píxeles)."""

    def __init__(self, carpeta_img: Path, carpeta_lab: Path | None = None, entrenar: bool = False,
                 fraccion: float = 1.0, semilla: int = 0):
        self.imgs = sorted(p for p in Path(carpeta_img).iterdir()
                           if p.suffix.lower() in EXT and not p.name.startswith("."))
        if fraccion < 1.0:
            random.Random(semilla).shuffle(self.imgs)
            self.imgs = sorted(self.imgs[:max(1, int(len(self.imgs) * fraccion))])
        self.lab, self.entrenar = carpeta_lab, entrenar

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        from torchvision.transforms.functional import to_tensor
        p = self.imgs[i]
        im = Image.open(p).convert("RGB")
        W, H = im.size
        x = to_tensor(im)
        cajas, clases = [], []
        if self.lab is not None:
            f = Path(self.lab) / (p.stem + ".txt")
            if f.exists():
                for l in f.read_text().splitlines():
                    v = l.split()
                    if len(v) >= 5:
                        c, xc, yc, w, h = int(float(v[0])), *map(float, v[1:5])
                        cajas.append([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H])
                        clases.append(c + 1)  # 0 es el fondo en torchvision
        b = torch.tensor(cajas, dtype=torch.float32).reshape(-1, 4)
        b[:, 0::2] = b[:, 0::2].clamp(0, W)
        b[:, 1::2] = b[:, 1::2].clamp(0, H)
        if self.entrenar:  # imágenes cenitales: no hay "arriba" ni "abajo"
            if random.random() < 0.5:
                x = x.flip(-1); b[:, [0, 2]] = W - b[:, [2, 0]]
            if random.random() < 0.5:
                x = x.flip(-2); b[:, [1, 3]] = H - b[:, [3, 1]]
        ok = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)
        objetivo = {"boxes": b[ok], "labels": torch.tensor(clases, dtype=torch.int64)[ok]}
        return x, objetivo, p, (W, H)


def juntar(lote):
    return tuple(zip(*lote))


# --------------------------------------------------------------------------- modelo
def construir_modelo(arquitectura: str, imgsz: int, preentrenado: bool = True):
    from torchvision.models import detection as D
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    opciones = {
        "resnet50_v2": (D.fasterrcnn_resnet50_fpn_v2, "FasterRCNN_ResNet50_FPN_V2_Weights"),
        "resnet50": (D.fasterrcnn_resnet50_fpn, "FasterRCNN_ResNet50_FPN_Weights"),
        "mobilenet": (D.fasterrcnn_mobilenet_v3_large_fpn, "FasterRCNN_MobileNet_V3_Large_FPN_Weights"),
    }
    fn, pesos = opciones[arquitectura]
    pesos = getattr(D, pesos).DEFAULT if preentrenado else None
    # min_size = max_size = imgsz -> el lado largo de cada imagen queda en imgsz (como imgsz en YOLO)
    m = fn(weights=pesos, min_size=imgsz, max_size=imgsz, box_detections_per_img=300)
    entrada = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(entrada, NC + 1)
    return m


def elegir_device(pedido: str) -> torch.device:
    if pedido != "auto":
        return torch.device(pedido if not pedido.isdigit() else f"cuda:{pedido}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Usando Apple Silicon (mps)")
        return torch.device("mps")
    print("⚠️  Sin GPU: el entrenamiento en CPU es MUY lento. Use Colab/Kaggle o --fraccion 0.05 --imgsz 640.")
    return torch.device("cpu")


def usa_amp(dev: torch.device) -> bool:
    # precisión mixta solo en GPUs NVIDIA con núcleos tensor (T4, A100, RTX...). En GPUs viejas es más lenta.
    return dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 7


# --------------------------------------------------------------------------- entrenamiento
def entrenar(m, dl, dev, epocas: int, horas: float | None, lr: float, carpeta_pesos: Path, meta: dict) -> Path:
    m.to(dev).train()
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    amp = usa_amp(dev)
    escalador = torch.cuda.amp.GradScaler(enabled=amp)
    limite = horas * 3600 if horas else None
    total = epocas * len(dl)
    calentamiento = min(300, total // 5 + 1)
    carpeta_pesos.mkdir(parents=True, exist_ok=True)
    ultimo = carpeta_pesos / "last.pt"
    t0, it, fin = time.time(), 0, False
    print(f"Entrenando: {len(dl.dataset)} imágenes · lote {dl.batch_size} · "
          f"{'límite ' + str(horas) + ' h' if limite else str(epocas) + ' épocas'} · precisión mixta: {amp}")
    for ep in range(1, epocas + 1):
        suma, n, t_ep = 0.0, 0, time.time()
        for imgs, objetivos, _, _ in dl:
            progreso = min(1.0, (time.time() - t0) / limite) if limite else it / total
            factor = min(1.0, (it + 1) / calentamiento) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progreso)))
            for g in opt.param_groups:
                g["lr"] = lr * factor
            imgs = [x.to(dev) for x in imgs]
            objetivos = [{k: v.to(dev) for k, v in o.items()} for o in objetivos]
            ctx = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
            with ctx:
                perdidas = m(imgs, objetivos)
                perdida = sum(perdidas.values())
            if not torch.isfinite(perdida):
                print("  pérdida no finita: se salta el lote"); opt.zero_grad(); it += 1; continue
            opt.zero_grad()
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            suma += float(perdida); n += 1; it += 1
            if it % 50 == 0:
                print(f"  época {ep} · iter {it} · pérdida {suma / n:.3f} · lr {lr * factor:.5f} · "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if limite and time.time() - t0 > limite:
                fin = True
                break
        torch.save({"modelo": m.state_dict(), **meta}, ultimo)
        print(f"Época {ep} terminada en {(time.time() - t_ep) / 60:.1f} min · pérdida media {suma / max(n, 1):.3f} "
              f"· pesos: {ultimo}", flush=True)
        if fin:
            print("Se alcanzó el límite de tiempo.")
            break
    return ultimo


# --------------------------------------------------------------------------- predicción
@torch.no_grad()
def predecir_carpeta(m, carpeta: Path, salida: Path, dev, lote: int = 4, workers: int = 2) -> int:
    """Escribe un .txt por imagen (aunque no tenga detecciones) en formato oficial."""
    m.to(dev).eval()
    m.roi_heads.score_thresh = 0.001   # enviar también las cajas de baja confianza (el mAP las aprovecha)
    m.roi_heads.detections_per_img = 300
    salida.mkdir(parents=True, exist_ok=True)
    ds = DatasetYOLO(carpeta)
    dl = torch.utils.data.DataLoader(ds, batch_size=lote, shuffle=False, num_workers=workers, collate_fn=juntar)
    amp = usa_amp(dev)
    n = 0
    for imgs, _, rutas, tams in dl:
        ctx = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
        with ctx:
            salidas = m([x.to(dev) for x in imgs])
        for o, ruta, (W, H) in zip(salidas, rutas, tams):
            dets = [(int(c) - 1, *b, s) for b, c, s in
                    zip(o["boxes"].float().tolist(), o["labels"].tolist(), o["scores"].float().tolist())]
            FE.escribir_txt(str(salida / (ruta.stem + ".txt")), dets, W, H)
            n += 1
        if n % 200 < lote:
            print(f"  {n}/{len(ds)} imágenes", flush=True)
    return n


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", required=True, help="carpeta spacelab_dataset")
    ap.add_argument("--arquitectura", default="resnet50_v2", choices=["resnet50_v2", "resnet50", "mobilenet"])
    ap.add_argument("--imgsz", type=int, default=1024, help="lado largo de la imagen al entrenar/predecir")
    ap.add_argument("--epocas", type=int, default=12)
    ap.add_argument("--horas", type=float, default=None, help="límite de tiempo de entrenamiento")
    ap.add_argument("--batch", type=int, default=None, help="imágenes por lote (defecto: 4 en GPU, 1 en mps/cpu)")
    ap.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje (defecto: 0.0025 × lote)")
    ap.add_argument("--fraccion", type=float, default=1.0, help="fracción de train a usar (pruebas rápidas)")
    ap.add_argument("--device", default="auto", help="auto | 0 | cuda | mps | cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--proyecto", default="runs_spacelab")
    ap.add_argument("--nombre", default="frcnn")
    ap.add_argument("--pesos", default=None, help="saltar el entrenamiento y usar estos pesos (last.pt)")
    a = ap.parse_args()

    datos = Path(a.datos).resolve()
    if not (datos / "images" / "train").exists():
        sys.exit(f"No encuentro {datos / 'images' / 'train'}")
    dev = elegir_device(a.device)
    torch.manual_seed(0); random.seed(0)
    salida = Path(a.proyecto).resolve() / a.nombre
    t0 = time.time()

    # ------------------------------------------------------------------ 1. entrenamiento
    if a.pesos:
        ck = torch.load(a.pesos, map_location="cpu")
        m = construir_modelo(ck.get("arquitectura", a.arquitectura), ck.get("imgsz", a.imgsz), preentrenado=False)
        m.load_state_dict(ck["modelo"])
    else:
        lote = a.batch or (4 if dev.type == "cuda" else 1)
        m = construir_modelo(a.arquitectura, a.imgsz)
        ds = DatasetYOLO(datos / "images" / "train", datos / "labels" / "train", entrenar=True, fraccion=a.fraccion)
        dl = torch.utils.data.DataLoader(ds, batch_size=lote, shuffle=True, num_workers=a.workers,
                                         collate_fn=juntar, drop_last=True, persistent_workers=a.workers > 0)
        entrenar(m, dl, dev, a.epocas, a.horas, a.lr or 0.0025 * lote, salida / "weights",
                 {"arquitectura": a.arquitectura, "imgsz": a.imgsz})
        print(f"\nEntrenamiento terminado en {(time.time() - t0) / 60:.1f} min")

    # ------------------------------------------------------------------ 2. puntaje local en test
    print("\nPrediciendo images/test para calcular el puntaje oficial local ...")
    pred_test = salida / "pred_test"
    predecir_carpeta(m, datos / "images" / "test", pred_test, dev, workers=a.workers)
    gt_test = EV.leer_gt(str(datos / "labels" / "test"))
    env_test = EV.leer_envio(str(pred_test))
    u, _ = EV.barrer_umbral(EV.preparar(gt_test, env_test))
    res = EV.evaluar(gt_test, env_test, EV.leer_meta(None, gt_test), umbral=u)
    EV.imprimir(res, "split test (local)")
    print(f" Umbral elegido (maximiza F1 en test): {u:.2f}")

    # ------------------------------------------------------------------ 3. envío sobre el conjunto oculto
    oculto = datos / "images" / "oculto"
    if not oculto.exists():
        print("No hay images/oculto; no se genera envío.")
        return 0
    print("\nPrediciendo images/oculto ...")
    pred_oc = salida / "envio"
    n = predecir_carpeta(m, oculto, pred_oc, dev, workers=a.workers)
    zpath = salida / f"envio_{a.nombre}.zip"
    FE.crear_zip(str(pred_oc), str(zpath), umbral=round(u, 2))
    print(f"\n✅ Envío listo: {zpath}  ({n} imágenes, umbral {u:.2f})")
    print(f"   Puntaje local esperado (test): {res['total']['score']:.4f}  ·  tiempo total {(time.time() - t0) / 60:.1f} min")
    print("   Revíselo con:  python formato_envio.py verificar --envio", zpath, "--imagenes", oculto)
    return 0


if __name__ == "__main__":
    sys.exit(main())
