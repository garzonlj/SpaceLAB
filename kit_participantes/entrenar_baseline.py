#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
entrenar_baseline.py — Propuesta base (baseline) · SpaceLAB Hackatón FAC+PUJ · GeoVision Colombia
=================================================================================================

Hace TODO el ciclo del reto con Ultralytics YOLO:
  1. Entrena un detector sobre images/train (valida durante el entrenamiento con images/test).
  2. Predice sobre images/test y calcula el PUNTAJE OFICIAL local con evaluar.py
     (0.7·mAP@0.5:0.95 + 0.3·F1macro@0.5) y busca el umbral de confianza que maximiza F1.
  3. Predice sobre images/oculto y genera el ZIP listo para subir al leaderboard
     (un .txt por imagen: `clase xc yc w h confianza` + umbral.txt).

Funciona en GPU NVIDIA (Colab / Kaggle T4), Apple Silicon (mps) o CPU (muy lento: use --fraccion).

Instalación:   pip install ultralytics
Ejemplos:
  # Colab/Kaggle T4, ~1.5 h de entrenamiento como máximo
  python entrenar_baseline.py --datos /content/spacelab_dataset --horas 1.5

  # Prueba rápida en un portátil (10 % de los datos, 3 épocas, imágenes de 640)
  python entrenar_baseline.py --datos ./spacelab_dataset --fraccion 0.1 --epocas 3 --imgsz 640

  # Solo generar el envío con unos pesos ya entrenados
  python entrenar_baseline.py --datos ./spacelab_dataset --pesos runs_spacelab/baseline/weights/best.pt

Ideas para superar el baseline (¡ese es el reto!): modelos más grandes (s/m), mayor resolución
(objetos diminutos), recortes/tiles de las imágenes grandes, augmentations para imágenes cenitales
(rotaciones, flip vertical), balanceo de la clase work_vessel, TTA, ensambles (WBF), más épocas...
Nota: si va a usar TTA, tenga en cuenta que con YOLO26 la opción `augment=True` de Ultralytics no tiene efecto
(la ignora con un aviso y predice normal).

Memoria: algunas imágenes tienen cientos de barcos y disparan el consumo de memoria al entrenar.
Si aparece `CUDA out of memory`, use un --batch pequeño (sobre todo con modelos grandes o mayor resolución).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import evaluar as EV  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def elegir_device(pedido: str):
    import torch
    if pedido != "auto":
        return pedido
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return 0
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Usando Apple Silicon (mps)")
        return "mps"
    print("⚠️  Sin GPU: el entrenamiento en CPU es MUY lento. Use Colab/Kaggle o --fraccion 0.05 --imgsz 640.")
    return "cpu"


def _predecir_por_lotes(modelo, imgs, lote: int = 8, **kw):
    """Devuelve (ruta, resultado) prediciendo de a `lote` imágenes.
    Ojo: si a Ultralytics se le pasa la lista completa de rutas, la carga entera como UN solo lote y se
    queda sin memoria (GPU o RAM); además, con listas pierde el nombre del archivo (r.path = 'image0.jpg'),
    por eso el nombre se toma de la ruta original y no de r.path."""
    for i in range(0, len(imgs), lote):
        parte = imgs[i:i + lote]
        yield from zip(parte, modelo.predict(source=[str(p) for p in parte], verbose=False, **kw))


def predecir_carpeta(modelo, carpeta: Path, salida: Path, imgsz: int, device, conf: float = 0.001) -> int:
    """Escribe un .txt por imagen (aunque no tenga detecciones) en formato oficial."""
    salida.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXT and not p.name.startswith("."))
    n = 0
    for ruta, r in _predecir_por_lotes(modelo, imgs, conf=conf, iou=0.6, imgsz=imgsz, max_det=300, device=device):
        b = r.boxes
        lineas = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.5f}"
                  for (x, y, w, h), c, s in zip(b.xywhn.tolist(), b.cls.tolist(), b.conf.tolist())]
        (salida / (ruta.stem + ".txt")).write_text("\n".join(lineas), encoding="utf-8")
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{len(imgs)} imágenes")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", required=True, help="carpeta spacelab_dataset (con data.yaml)")
    ap.add_argument("--modelo", default="yolo26n.pt", help="pesos preentrenados (yolo26n/s/m.pt, yolo11n.pt, ...)")
    ap.add_argument("--imgsz", type=int, default=1024, help="resolución de entrenamiento/predicción")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--horas", type=float, default=None, help="límite de tiempo de entrenamiento (sobrescribe épocas)")
    ap.add_argument("--batch", type=float, default=None, help="tamaño de lote (defecto: auto en GPU, 4 en mps/cpu); si se queda sin memoria, use uno pequeño")
    ap.add_argument("--fraccion", type=float, default=1.0, help="fracción de train a usar (pruebas rápidas)")
    ap.add_argument("--device", default="auto", help="auto | 0 | mps | cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--proyecto", default="runs_spacelab")
    ap.add_argument("--nombre", default="baseline")
    ap.add_argument("--pesos", default=None, help="saltar el entrenamiento y usar estos pesos")
    a = ap.parse_args()

    from ultralytics import YOLO

    datos = Path(a.datos).resolve()
    yaml = datos / "data.yaml"
    if not yaml.exists():
        sys.exit(f"No existe {yaml}")
    device = elegir_device(a.device)
    t0 = time.time()

    # ------------------------------------------------------------------ 1. entrenamiento
    if a.pesos:
        pesos = Path(a.pesos)
    else:
        batch = a.batch if a.batch is not None else (-1 if device == 0 else 4)
        modelo = YOLO(a.modelo)
        extra = {"time": a.horas} if a.horas else {}
        modelo.train(
            data=str(yaml), epochs=a.epocas, imgsz=a.imgsz, batch=int(batch) if batch >= 1 or batch == -1 else batch,
            device=device, workers=a.workers, fraction=a.fraccion, project=str(Path(a.proyecto).resolve()),
            name=a.nombre, exist_ok=True, patience=15, cos_lr=True, close_mosaic=5, seed=0,
            flipud=0.5, fliplr=0.5,  # imágenes cenitales: no hay "arriba" ni "abajo"
            plots=True, **extra)
        pesos = Path(modelo.trainer.best)
        print(f"\nEntrenamiento terminado en {(time.time() - t0) / 60:.1f} min · mejores pesos: {pesos}")

    modelo = YOLO(str(pesos))
    salida = Path(a.proyecto).resolve() / a.nombre

    # ------------------------------------------------------------------ 2. puntaje local en test
    print("\nPrediciendo images/test para calcular el puntaje oficial local ...")
    pred_test = salida / "pred_test"
    predecir_carpeta(modelo, datos / "images" / "test", pred_test, a.imgsz, device)
    gt_test = EV.leer_gt(str(datos / "labels" / "test"))
    env_test = EV.leer_envio(str(pred_test))
    u, f1 = EV.barrer_umbral(EV.preparar(gt_test, env_test))
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
    n = predecir_carpeta(modelo, oculto, pred_oc, a.imgsz, device)
    (pred_oc / "umbral.txt").write_text(f"{u:.2f}\n", encoding="utf-8")
    zpath = salida / f"envio_{a.nombre}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pred_oc.glob("*.txt")):
            z.write(p, p.name)
    print(f"\n✅ Envío listo: {zpath}  ({n} imágenes, umbral {u:.2f})")
    print(f"   Puntaje local esperado (test): {res['total']['score']:.4f}  ·  tiempo total {(time.time() - t0) / 60:.1f} min")
    print("   Súbalo en la página del leaderboard (sección 📤 Enviar).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
