#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluar.py — Evaluador oficial · SpaceLAB Hackatón FAC+PUJ · GeoVision Colombia
================================================================================

Puntaje oficial
---------------
    Score = 0.7 · mAP@0.5:0.95  +  0.3 · F1macro@0.5

* mAP@0.5:0.95  Promedio (macro) sobre las 5 clases del AP estilo COCO, calculado en
                10 umbrales de IoU (0.50, 0.55, ..., 0.95) con interpolación de 101 puntos.
                Usa TODAS las cajas enviadas (máximo 300 por imagen, las de mayor confianza).
* F1macro@0.5   Promedio (macro) sobre las 5 clases del F1 con IoU ≥ 0.5, usando SOLO las
                cajas con confianza ≥ umbral del equipo (archivo `umbral.txt` dentro del envío;
                por defecto 0.25). Mide qué tan "desplegable" es el sistema: cuántos barcos
                encuentra y cuántas falsas alarmas produce.
* Desempate     mAP@0.5 y, si persiste, el envío más temprano.

Formato del envío
-----------------
Un ZIP (o una carpeta) con un archivo .txt por imagen, con el MISMO nombre de la imagen
(p. ej. `sl_3fa9c2d1.jpg` -> `sl_3fa9c2d1.txt`). Una línea por detección, formato YOLO:

    clase x_centro y_centro ancho alto confianza

con coordenadas normalizadas entre 0 y 1 (relativas al ancho/alto de la imagen) y clase
entera 0..4 (0 small_boat, 1 cargo_ship, 2 fishing_vessel, 3 warship, 4 work_vessel).
Es exactamente lo que produce Ultralytics con `save_txt=True, save_conf=True`.
Imágenes sin archivo = sin detecciones. Opcional: `umbral.txt` con un número (p. ej. 0.30).

Uso
---
  # Participantes: evaluar sus predicciones sobre el split `test` (que tiene etiquetas)
  python evaluar.py --gt spacelab_dataset/labels/test --envio mis_predicciones.zip

  # Organizadores: un envío contra el conjunto oculto (público / privado)
  python evaluar.py --gt privado/gt_oculto/labels --meta privado/gt_oculto/meta.csv --envio equipo.zip

  # Organizadores: ranking de todos los envíos de una carpeta (un ZIP por equipo)
  python evaluar.py --gt privado/gt_oculto/labels --meta privado/gt_oculto/meta.csv \
                    --envios carpeta_envios/ --ranking ranking.csv

Solo depende de numpy.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import zipfile
from dataclasses import dataclass, field

import numpy as np

CLASES = ["small_boat", "cargo_ship", "fishing_vessel", "warship", "work_vessel"]
NC = len(CLASES)
UMBRALES_IOU = np.linspace(0.5, 0.95, 10)
PUNTOS_RECALL = np.linspace(0.0, 1.0, 101)
MAX_DET = 300
UMBRAL_DEFECTO = 0.25
PESO_MAP, PESO_F1 = 0.7, 0.3


# --------------------------------------------------------------------------- lectura
@dataclass
class Envio:
    preds: dict            # id_imagen -> np.ndarray (N, 6): cls, xc, yc, w, h, conf
    umbral: float = UMBRAL_DEFECTO
    umbral_declarado: bool = False
    avisos: list = field(default_factory=list)


def _parse_lineas(texto: str, nombre: str, con_conf: bool, avisos: list) -> np.ndarray:
    filas = []
    for i, linea in enumerate(texto.splitlines(), 1):
        p = linea.replace(",", " ").split()
        if not p:
            continue
        if len(p) not in (5, 6):
            avisos.append(f"{nombre}:{i}: se esperaban 6 valores, hay {len(p)} -> línea ignorada")
            continue
        try:
            c = int(float(p[0]))
            xc, yc, w, h = (float(v) for v in p[1:5])
            conf = float(p[5]) if len(p) == 6 else 1.0
        except ValueError:
            avisos.append(f"{nombre}:{i}: valores no numéricos -> línea ignorada")
            continue
        if len(p) == 5 and con_conf:
            avisos.append(f"{nombre}: líneas sin confianza; se asume 1.0 (perjudica el mAP)")
            con_conf = False  # avisar solo una vez por archivo
        if not 0 <= c < NC:
            avisos.append(f"{nombre}:{i}: clase {c} fuera de rango 0..{NC-1} -> línea ignorada")
            continue
        if not all(np.isfinite([xc, yc, w, h, conf])):
            avisos.append(f"{nombre}:{i}: valores no finitos -> línea ignorada")
            continue
        if w <= 0 or h <= 0:  # cajas degeneradas: se ignoran y se reportan en un solo aviso
            avisos.append("__degenerada__")
            continue
        filas.append((c, xc, yc, w, h, conf))
    return np.asarray(filas, dtype=np.float64).reshape(-1, 6)


def _es_basura(ruta: str) -> bool:
    base = os.path.basename(ruta)
    return ("__MACOSX" in ruta or base.startswith("._") or base.startswith(".")
            or not base.lower().endswith(".txt"))


def leer_envio(ruta: str) -> Envio:
    """Lee un envío desde un .zip o una carpeta. Busca .txt en cualquier subcarpeta."""
    archivos = {}  # nombre_base -> texto
    if os.path.isdir(ruta):
        for raiz, _, fs in os.walk(ruta):
            for f in fs:
                r = os.path.join(raiz, f)
                if not _es_basura(r):
                    with open(r, "r", encoding="utf-8", errors="replace") as fh:
                        archivos.setdefault(f, fh.read())
    elif zipfile.is_zipfile(ruta):
        with zipfile.ZipFile(ruta) as z:
            for info in z.infolist():
                if info.is_dir() or _es_basura(info.filename):
                    continue
                if info.file_size > 50_000_000:
                    raise ValueError(f"Archivo demasiado grande dentro del ZIP: {info.filename}")
                archivos.setdefault(os.path.basename(info.filename),
                                    z.read(info).decode("utf-8", errors="replace"))
    else:
        raise ValueError(f"No es un ZIP ni una carpeta: {ruta}")

    env = Envio(preds={})
    if "umbral.txt" in archivos:
        try:
            u = float(archivos.pop("umbral.txt").strip().split()[0])
            if not 0.0 <= u <= 1.0:
                raise ValueError
            env.umbral, env.umbral_declarado = u, True
        except (ValueError, IndexError):
            env.avisos.append("umbral.txt inválido; se usa el umbral por defecto 0.25")
    for nombre, texto in archivos.items():
        env.preds[os.path.splitext(nombre)[0]] = _parse_lineas(texto, nombre, True, env.avisos)
    n_deg = env.avisos.count("__degenerada__")
    env.avisos = [a for a in env.avisos if a != "__degenerada__"]
    if n_deg:
        env.avisos.insert(0, f"{n_deg} cajas con ancho o alto ≤ 0 fueron ignoradas")
    if not env.preds:
        raise ValueError("El envío no contiene archivos .txt de predicciones.")
    return env


def leer_gt(carpeta: str) -> dict:
    """Etiquetas YOLO de referencia: id_imagen -> np.ndarray (M, 5): cls, xc, yc, w, h."""
    gt = {}
    for f in sorted(os.listdir(carpeta)):
        if _es_basura(f):
            continue
        with open(os.path.join(carpeta, f), "r", encoding="utf-8") as fh:
            filas = [list(map(float, l.split()[:5])) for l in fh if len(l.split()) >= 5]
        gt[os.path.splitext(f)[0]] = np.asarray(filas, dtype=np.float64).reshape(-1, 5)
    if not gt:
        raise ValueError(f"No se encontraron etiquetas en {carpeta}")
    return gt


def leer_meta(ruta: str | None, ids) -> dict:
    """id -> {'fuente':..., 'particion':...}. Sin meta: fuente = prefijo antes de '__'."""
    meta = {}
    if ruta:
        with open(ruta, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                meta[r["id"]] = {"fuente": r.get("fuente", ""), "particion": r.get("particion", "")}
    for i in ids:
        meta.setdefault(i, {"fuente": i.split("__")[0] if "__" in i else "", "particion": ""})
    return meta


# --------------------------------------------------------------------------- núcleo
def _xyxy(b: np.ndarray) -> np.ndarray:
    xc, yc, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], 1)


def iou_matriz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre cajas xc,yc,w,h normalizadas (la IoU es invariante al escalado por eje)."""
    A, B = _xyxy(a), _xyxy(b)
    ix = np.clip(np.minimum(A[:, None, 2], B[None, :, 2]) - np.maximum(A[:, None, 0], B[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(A[:, None, 3], B[None, :, 3]) - np.maximum(A[:, None, 1], B[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-12)


def _emparejar(ious: np.ndarray) -> np.ndarray:
    """Emparejamiento voraz estilo COCO (detecciones ya ordenadas por confianza desc.).
    Devuelve (T, n_det) bool: True si la detección es TP en cada umbral de IoU."""
    T = len(UMBRALES_IOU)
    n_det, n_gt = ious.shape
    tp = np.zeros((T, n_det), dtype=bool)
    if n_det == 0 or n_gt == 0:
        return tp
    usado = np.zeros((T, n_gt), dtype=bool)
    mejor = ious.max(1)
    thr = UMBRALES_IOU[:, None]
    filas = np.arange(T)
    for d in np.nonzero(mejor >= UMBRALES_IOU[0] - 1e-12)[0]:
        cand = np.where((ious[d][None, :] >= thr - 1e-12) & ~usado, ious[d][None, :], -1.0)
        g = cand.argmax(1)
        ok = cand[filas, g] >= 0
        tp[ok, d] = True
        usado[filas[ok], g[ok]] = True
    return tp


def _ap(tp_ord: np.ndarray, n_gt: int) -> np.ndarray:
    """AP COCO (101 puntos) para cada umbral. tp_ord: (T, N) ordenado por confianza desc."""
    T = tp_ord.shape[0]
    if n_gt == 0:
        return np.full(T, np.nan)
    if tp_ord.shape[1] == 0:
        return np.zeros(T)
    tpc = np.cumsum(tp_ord, 1, dtype=np.float64)
    fpc = np.cumsum(~tp_ord, 1, dtype=np.float64)
    rec = tpc / n_gt
    pre = tpc / np.maximum(tpc + fpc, np.finfo(np.float64).eps)
    pre = np.flip(np.maximum.accumulate(np.flip(pre, 1), 1), 1)  # envolvente monótona
    out = np.zeros(T)
    for t in range(T):
        idx = np.searchsorted(rec[t], PUNTOS_RECALL, side="left")
        q = np.zeros(len(PUNTOS_RECALL))
        v = idx < len(rec[t])
        q[v] = pre[t][idx[v]]
        out[t] = q.mean()
    return out


@dataclass
class _PorClase:
    conf: list = field(default_factory=list)
    tp: list = field(default_factory=list)     # (T, n) por imagen
    img: list = field(default_factory=list)    # id de imagen por detección
    n_gt_img: dict = field(default_factory=dict)


def preparar(gt: dict, env: Envio) -> list:
    """Empareja una sola vez por imagen y clase; luego se agregan subconjuntos baratos."""
    pc = [_PorClase() for _ in range(NC)]
    for img, g in gt.items():
        p = env.preds.get(img, np.zeros((0, 6)))
        if len(p) > MAX_DET:
            p = p[np.argsort(-p[:, 5], kind="mergesort")[:MAX_DET]]
        for c in range(NC):
            gc = g[g[:, 0] == c, 1:5]
            pcd = p[p[:, 0] == c]
            pcd = pcd[np.argsort(-pcd[:, 5], kind="mergesort")]
            pc[c].n_gt_img[img] = len(gc)
            if len(pcd) == 0:
                continue
            ious = iou_matriz(pcd[:, 1:5], gc) if len(gc) else np.zeros((len(pcd), 0))
            pc[c].conf.append(pcd[:, 5])
            pc[c].tp.append(_emparejar(ious))
            pc[c].img.extend([img] * len(pcd))
    for c in range(NC):
        k = pc[c]
        k.conf = np.concatenate(k.conf) if k.conf else np.zeros(0)
        k.tp = np.concatenate(k.tp, 1) if k.tp else np.zeros((len(UMBRALES_IOU), 0), bool)
        k.img = np.asarray(k.img, dtype=object)
    return pc


def metricas(pc: list, umbral: float, subconjunto: set | None = None) -> dict:
    ap = np.full((NC, len(UMBRALES_IOU)), np.nan)
    P, R, F1, n_gt_c = np.zeros(NC), np.zeros(NC), np.zeros(NC), np.zeros(NC, int)
    for c, k in enumerate(pc):
        if subconjunto is None:
            m = np.ones(len(k.conf), bool)
            n_gt = sum(k.n_gt_img.values())
        else:
            m = np.fromiter((i in subconjunto for i in k.img), bool, len(k.img))
            n_gt = sum(v for i, v in k.n_gt_img.items() if i in subconjunto)
        conf, tp = k.conf[m], k.tp[:, m]
        orden = np.argsort(-conf, kind="mergesort")
        ap[c] = _ap(tp[:, orden], n_gt)
        sel = conf >= umbral
        n_tp = int(tp[0, sel].sum()); n_fp = int(sel.sum()) - n_tp; n_fn = n_gt - n_tp
        P[c] = n_tp / (n_tp + n_fp) if n_tp + n_fp else 0.0
        R[c] = n_tp / n_gt if n_gt else 0.0
        F1[c] = 2 * n_tp / (2 * n_tp + n_fp + n_fn) if (2 * n_tp + n_fp + n_fn) else 0.0
        n_gt_c[c] = n_gt
    validas = n_gt_c > 0
    map5095 = float(np.nanmean(ap[validas].mean(1))) if validas.any() else 0.0
    map50 = float(np.nanmean(ap[validas, 0])) if validas.any() else 0.0
    f1m = float(F1[validas].mean()) if validas.any() else 0.0
    return {
        "score": PESO_MAP * map5095 + PESO_F1 * f1m,
        "map50_95": map5095, "map50": map50, "f1_macro": f1m,
        "precision_macro": float(P[validas].mean()) if validas.any() else 0.0,
        "recall_macro": float(R[validas].mean()) if validas.any() else 0.0,
        "por_clase": {CLASES[c]: {"ap50_95": float(np.nan_to_num(ap[c].mean())), "ap50": float(np.nan_to_num(ap[c, 0])),
                                  "precision": float(P[c]), "recall": float(R[c]), "f1": float(F1[c]),
                                  "n_gt": int(n_gt_c[c])} for c in range(NC)},
    }


def barrer_umbral(pc: list, umbrales=None) -> tuple[float, float]:
    """Umbral de confianza que maximiza F1macro@0.5 (útil para elegir `umbral.txt` con el split test)."""
    umbrales = np.round(np.arange(0.05, 0.91, 0.01), 2) if umbrales is None else np.asarray(umbrales)
    f1s = []
    for k in pc:
        n_gt = sum(k.n_gt_img.values())
        if n_gt == 0:
            continue
        orden = np.argsort(-k.conf, kind="mergesort")
        conf, tp = k.conf[orden], k.tp[0, orden]
        tpc = np.concatenate([[0], np.cumsum(tp)])
        n_sel = np.searchsorted(-conf, -umbrales, side="right")   # nº de cajas con conf >= u
        n_tp = tpc[n_sel]; n_fp = n_sel - n_tp; n_fn = n_gt - n_tp
        f1s.append(2 * n_tp / np.maximum(2 * n_tp + n_fp + n_fn, 1))
    f1m = np.mean(f1s, 0)
    i = int(np.argmax(f1m))
    return float(umbrales[i]), float(f1m[i])


def evaluar(gt: dict, env: Envio, meta: dict | None = None, umbral: float | None = None) -> dict:
    u = env.umbral if umbral is None else umbral
    ids_env = set(env.preds)
    ids_gt = set(gt)
    pc = preparar(gt, env)
    res = {"umbral": u, "umbral_declarado": env.umbral_declarado or umbral is not None,
           "imagenes_gt": len(ids_gt), "imagenes_con_pred": len(ids_env & ids_gt),
           "archivos_desconocidos": len(ids_env - ids_gt),
           "total": metricas(pc, u)}
    if meta:
        parts = sorted({m["particion"] for i, m in meta.items() if i in gt and m["particion"]})
        for p in parts:
            res[p] = metricas(pc, u, {i for i in gt if meta[i]["particion"] == p})
        fuentes = sorted({m["fuente"] for i, m in meta.items() if i in gt and m["fuente"]})
        res["por_fuente"] = {f: metricas(pc, u, {i for i in gt if meta[i]["fuente"] == f}) for f in fuentes}
    avisos = list(env.avisos)
    if res["archivos_desconocidos"]:
        avisos.insert(0, f"{res['archivos_desconocidos']} archivos no corresponden a ninguna imagen evaluada (ignorados)")
    if res["imagenes_con_pred"] == 0:
        avisos.insert(0, "Ningún archivo coincide con las imágenes evaluadas: ¿nombres de archivo correctos?")
    res["avisos"] = avisos[:50] + ([f"... y {len(avisos) - 50} avisos más"] if len(avisos) > 50 else [])
    return res


# --------------------------------------------------------------------------- salida
def imprimir(res: dict, titulo: str = "") -> None:
    t = res["total"]
    print("=" * 72)
    print(f" SpaceLAB · Evaluación {titulo}".rstrip())
    print("=" * 72)
    print(f" Imágenes evaluadas: {res['imagenes_gt']}  ·  con predicciones: {res['imagenes_con_pred']}"
          f"  ·  umbral F1: {res['umbral']:.2f}{'' if res['umbral_declarado'] else ' (por defecto)'}")
    for clave in ("publico", "privado", "total"):
        if clave in res:
            m = res[clave]
            print(f" {clave.upper():8s}  SCORE {m['score']:.4f}   mAP50-95 {m['map50_95']:.4f}   "
                  f"mAP50 {m['map50']:.4f}   F1 {m['f1_macro']:.4f}")
    print("-" * 72)
    print(f" {'clase':16s} {'AP50-95':>8s} {'AP50':>8s} {'P':>7s} {'R':>7s} {'F1':>7s} {'n_gt':>7s}")
    for c, m in t["por_clase"].items():
        print(f" {c:16s} {m['ap50_95']:8.4f} {m['ap50']:8.4f} {m['precision']:7.3f} {m['recall']:7.3f} {m['f1']:7.3f} {m['n_gt']:7d}")
    if res.get("por_fuente"):
        print("-" * 72)
        print(f" {'fuente':16s} {'SCORE':>8s} {'mAP50-95':>9s} {'F1':>7s}")
        for f, m in res["por_fuente"].items():
            print(f" {f:16s} {m['score']:8.4f} {m['map50_95']:9.4f} {m['f1_macro']:7.4f}")
    if res["avisos"]:
        print("-" * 72)
        for a in res["avisos"][:15]:
            print(" AVISO:", a)
    print("=" * 72)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluador oficial SpaceLAB (Score = 0.7·mAP50-95 + 0.3·F1@0.5)")
    ap.add_argument("--gt", required=True, help="carpeta con etiquetas YOLO de referencia (.txt)")
    ap.add_argument("--meta", help="CSV con columnas id,fuente,particion (conjunto oculto)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--envio", help="ZIP o carpeta con las predicciones de un equipo")
    g.add_argument("--envios", help="carpeta con un ZIP (o subcarpeta) por equipo -> ranking")
    ap.add_argument("--umbral", type=float, help="forzar umbral de confianza para F1 (ignora umbral.txt)")
    ap.add_argument("--barrer-umbral", action="store_true", help="buscar el umbral que maximiza F1 (usar con test)")
    ap.add_argument("--json", help="guardar resultado completo en JSON")
    ap.add_argument("--ranking", help="CSV de salida del ranking (con --envios)")
    a = ap.parse_args(argv)

    gt = leer_gt(a.gt)
    meta = leer_meta(a.meta, gt) if a.meta else leer_meta(None, gt)

    if a.envio:
        env = leer_envio(a.envio)
        res = evaluar(gt, env, meta, a.umbral)
        if a.barrer_umbral:
            u, f = barrer_umbral(preparar(gt, env))
            res["umbral_optimo"] = {"umbral": u, "f1_macro": f}
        imprimir(res, os.path.basename(a.envio.rstrip("/")))
        if a.barrer_umbral:
            print(f" Umbral que maximiza F1macro@0.5: {res['umbral_optimo']['umbral']:.2f} "
                  f"(F1 = {res['umbral_optimo']['f1_macro']:.4f}) -> escríbalo en umbral.txt")
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump(res, fh, ensure_ascii=False, indent=2)
        return 0

    filas = []
    for nombre in sorted(os.listdir(a.envios)):
        r = os.path.join(a.envios, nombre)
        if nombre.startswith(".") or not (os.path.isdir(r) or nombre.lower().endswith(".zip")):
            continue
        equipo = os.path.splitext(nombre)[0]
        try:
            res = evaluar(gt, leer_envio(r), meta, a.umbral)
        except Exception as e:  # noqa: BLE001
            print(f"[{equipo}] ERROR: {e}", file=sys.stderr)
            continue
        fila = {"equipo": equipo}
        for clave in ("total", "publico", "privado"):
            if clave in res:
                for k in ("score", "map50_95", "map50", "f1_macro"):
                    fila[f"{clave}_{k}"] = round(res[clave][k], 5)
        filas.append(fila)
        print(f"[{equipo}] total score {res['total']['score']:.4f}")
    clave = "privado_score" if filas and "privado_score" in filas[0] else "total_score"
    desempate = clave.replace("score", "map50")
    filas.sort(key=lambda f: (-f[clave], -f[desempate]))
    print("\n RANKING (" + clave + ")")
    for i, f in enumerate(filas, 1):
        print(f" {i:2d}. {f['equipo']:30s} {f[clave]:.4f}")
    if a.ranking and filas:
        with open(a.ranking, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader(); w.writerows(filas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
