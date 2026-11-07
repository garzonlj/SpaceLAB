#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
formato_envio.py — Ayuda para el formato de envío · SpaceLAB Hackatón FAC+PUJ · GeoVision Colombia
===================================================================================================

Sirve con CUALQUIER detector (YOLO de cualquier versión, Faster R-CNN, RT-DETR, Detectron2, MMDetection...).
Solo se encarga del formato: no toca su modelo ni usa las respuestas del conjunto oculto.

1) Desde su código (las cajas en PÍXELES, formato x1 y1 x2 y2, como las entregan casi todos los frameworks):

    from formato_envio import escribir_txt, crear_zip
    # detecciones de UNA imagen: lista de (clase, x1, y1, x2, y2, confianza), clase entera 0..4
    escribir_txt("preds/sl_3fa9c2d1.txt", detecciones, ancho_imagen, alto_imagen)
    ...
    crear_zip("preds/", "mi_envio.zip", umbral=0.30)

2) Desde la terminal:

    # arma el ZIP con los .txt de una carpeta (+ umbral.txt)
    python formato_envio.py zip --carpeta preds/ --salida mi_envio.zip --umbral 0.30

    # revisa un envío ANTES de subirlo: nombres, cantidad de archivos y formato de cada línea
    python formato_envio.py verificar --envio mi_envio.zip --imagenes spacelab_dataset/images/oculto

Formato oficial (lo que escribe este archivo): un .txt por imagen, mismo nombre que la imagen, una línea por
detección `clase x_centro y_centro ancho alto confianza`, coordenadas normalizadas 0-1. Máximo 300 por imagen.
La verificación usa el mismo lector que el evaluador oficial (evaluar.py, que debe estar en la misma carpeta).
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile

CLASES = ["small_boat", "cargo_ship", "fishing_vessel", "warship", "work_vessel"]
MAX_DET = 300
EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------- escribir
def linea(clase: int, x1: float, y1: float, x2: float, y2: float, conf: float, ancho: float, alto: float) -> str | None:
    """Una detección en píxeles (x1,y1,x2,y2) -> línea oficial. Devuelve None si la caja queda vacía."""
    x1, x2 = sorted((min(max(float(x1), 0.0), ancho), min(max(float(x2), 0.0), ancho)))
    y1, y2 = sorted((min(max(float(y1), 0.0), alto), min(max(float(y2), 0.0), alto)))
    w, h = (x2 - x1) / ancho, (y2 - y1) / alto
    if w <= 0 or h <= 0:
        return None
    c = int(clase)
    if not 0 <= c < len(CLASES):
        raise ValueError(f"clase {c} fuera de rango 0..{len(CLASES) - 1}")
    conf = min(max(float(conf), 0.0), 1.0)
    return f"{c} {(x1 + x2) / 2 / ancho:.6f} {(y1 + y2) / 2 / alto:.6f} {w:.6f} {h:.6f} {conf:.5f}"


def escribir_txt(ruta_txt: str, detecciones, ancho: float, alto: float) -> int:
    """Escribe el .txt de una imagen. detecciones: iterable de (clase, x1, y1, x2, y2, confianza) en píxeles.
    Se guardan como máximo las 300 de mayor confianza. Crea el archivo aunque no haya detecciones."""
    dets = sorted(detecciones, key=lambda d: -float(d[5]))[:MAX_DET]
    lineas = [l for l in (linea(*d, ancho, alto) for d in dets) if l]
    os.makedirs(os.path.dirname(os.path.abspath(ruta_txt)), exist_ok=True)
    with open(ruta_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas))
    return len(lineas)


def crear_zip(carpeta: str, ruta_zip: str, umbral: float | None = None) -> int:
    """Empaqueta todos los .txt de `carpeta` (sin subcarpetas) y, si se indica, umbral.txt."""
    txts = sorted(f for f in os.listdir(carpeta) if f.endswith(".txt") and f != "umbral.txt" and not f.startswith("."))
    if not txts:
        raise ValueError(f"No hay archivos .txt en {carpeta}")
    with zipfile.ZipFile(ruta_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in txts:
            z.write(os.path.join(carpeta, f), f)
        if umbral is not None:
            if not 0 <= umbral <= 1:
                raise ValueError("el umbral debe estar entre 0 y 1")
            z.writestr("umbral.txt", f"{umbral:.2f}\n")
    return len(txts)


# --------------------------------------------------------------------------- verificar
def verificar(envio: str, carpeta_imagenes: str | None = None) -> bool:
    """Revisa un envío (ZIP o carpeta) con el mismo lector del evaluador oficial. Devuelve True si se puede subir."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import evaluar as EV
    except ImportError:
        sys.exit("Para verificar, ponga evaluar.py en la misma carpeta que formato_envio.py")
    ok = True
    try:
        env = EV.leer_envio(envio)
    except ValueError as e:
        print(f"✗ {e}")
        return False
    n_det = sum(len(v) for v in env.preds.values())
    print(f"Archivos de predicción: {len(env.preds)} · detecciones válidas: {n_det}")
    print(f"Umbral para F1: {env.umbral:.2f} ({'umbral.txt' if env.umbral_declarado else 'no hay umbral.txt: se usará 0.25'})")
    muchos = [k for k, v in env.preds.items() if len(v) > MAX_DET]
    if muchos:
        print(f"! {len(muchos)} archivos tienen más de {MAX_DET} detecciones: solo cuentan las {MAX_DET} de mayor confianza")
    if carpeta_imagenes:
        imgs = {os.path.splitext(f)[0] for f in os.listdir(carpeta_imagenes)
                if os.path.splitext(f)[1].lower() in EXT_IMG and not f.startswith(".")}
        sobran = sorted(set(env.preds) - imgs)
        faltan = sorted(imgs - set(env.preds))
        print(f"Imágenes en {carpeta_imagenes}: {len(imgs)} · con archivo: {len(imgs & set(env.preds))}")
        if sobran:
            ok = False
            print(f"✗ {len(sobran)} archivos no corresponden a ninguna imagen (revise los nombres), p. ej.: {', '.join(sobran[:5])}")
        if faltan:
            print(f"! {len(faltan)} imágenes sin archivo: se califican como 'sin detecciones', p. ej.: {', '.join(faltan[:5])}")
        if not imgs & set(env.preds):
            ok = False
            print("✗ Ningún archivo coincide con las imágenes: el envío sería rechazado.")
    for a in env.avisos[:15]:
        print(f"! {a}")
    if len(env.avisos) > 15:
        print(f"! ... y {len(env.avisos) - 15} avisos más")
    print("✔ El envío se puede subir." if ok else "✗ Corrija lo anterior antes de subir.")
    return ok


# --------------------------------------------------------------------------- terminal
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zip", help="empaquetar una carpeta de .txt")
    z.add_argument("--carpeta", required=True); z.add_argument("--salida", required=True)
    z.add_argument("--umbral", type=float, default=None)
    v = sub.add_parser("verificar", help="revisar un envío antes de subirlo")
    v.add_argument("--envio", required=True, help="ZIP o carpeta")
    v.add_argument("--imagenes", default=None, help="carpeta images/oculto (para revisar los nombres)")
    a = ap.parse_args()
    if a.cmd == "zip":
        n = crear_zip(a.carpeta, a.salida, a.umbral)
        print(f"✔ {a.salida}: {n} archivos" + (f" + umbral.txt ({a.umbral:.2f})" if a.umbral is not None else ""))
        return 0
    return 0 if verificar(a.envio, a.imagenes) else 1


if __name__ == "__main__":
    sys.exit(main())
