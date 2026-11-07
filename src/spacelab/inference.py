"""Prediction at original coordinates, including class-aware tile fusion."""
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.ops import batched_nms

from .dataset import letterbox, verify_record


def _window_positions(length, size, overlap):
    if not 0 <= overlap < 1 or size <= 0:
        raise ValueError("Tile positivo y overlap en [0,1)")
    last = max(0,length-size)
    return sorted(set([*range(0,last+1,max(1,int(size*(1-overlap)))),last]))


def predict_image(detector, image, config, deadline):
    """Return [class,x1,y1,x2,y2,score], original pixels. Never resize source files."""
    width,height = image.size
    tiles = config.get("tile_size")
    xs = _window_positions(width,tiles,config.get("tile_overlap",.25)) if tiles else [0]
    ys = _window_positions(height,tiles,config.get("tile_overlap",.25)) if tiles else [0]
    predictions = []
    for x in xs:
        for y in ys:
            deadline.check(5)
            crop = image.crop((x,y,min(width,x+tiles),min(height,y+tiles))) if tiles else image
            canvas,(sx,sy,dx,dy) = letterbox(crop,config["imgsz"])
            tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2,0,1).float()[None]/255
            tensor = tensor.to(next(detector.net.parameters()).device)
            try:
                with torch.autocast("cuda",dtype=torch.float16,enabled=config.get("amp",True)):
                    result = detector.predict(tensor,nms_iou=config["nms_iou"])[0]
            except torch.cuda.OutOfMemoryError:
                del tensor
                torch.cuda.empty_cache()
                # No hidden resolution or confidence change; caller records interruption.
                raise RuntimeError("OOM con inferencia batch=1. Calibrar otra configuración explícita.")
            result[:,[1,3]] = (result[:,[1,3]]-dx)/sx+x
            result[:,[2,4]] = (result[:,[2,4]]-dy)/sy+y
            result[:,[1,3]] = result[:,[1,3]].clip(0,width)
            result[:,[2,4]] = result[:,[2,4]].clip(0,height)
            result = result[(result[:,3]>result[:,1])&(result[:,4]>result[:,2])]
            predictions.append(result)
    merged = np.concatenate(predictions) if predictions else np.zeros((0,6))
    if tiles and len(merged):
        keep = batched_nms(torch.tensor(merged[:,1:5]),torch.tensor(merged[:,5]),
                           torch.tensor(merged[:,0]),config.get("tile_merge_iou",.5)).numpy()
        merged = merged[keep]
    merged = merged[np.argsort(-merged[:,5],kind="stable")[:300]]
    return merged


def predictions(detector, root, records, config, deadline, destination):
    from .evaluation import official_modules
    _, fe = official_modules()
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    data, t0 = {}, time.perf_counter()
    for index, row in enumerate(records, 1):
        deadline.check(5)
        verify_record(root,row)
        with Image.open(Path(root)/row["image"]) as im:
            image = im.convert("RGB")
        w,h = image.size
        p = predict_image(detector,image,config,deadline)
        normalized = np.column_stack([p[:,0],(p[:,1]+p[:,3])/2/w,(p[:,2]+p[:,4])/2/h,
                                     (p[:,3]-p[:,1])/w,(p[:,4]-p[:,2])/h,p[:,5]])
        data[row["id"]] = normalized
        # Official writer uses native-pixel xyxy boxes and normalizes itself.
        fe.escribir_txt(str(destination/(row["id"]+".txt")),p,w,h)
        if index % 50 == 0 or index == len(records):
            print(f"Inferencia: {index}/{len(records)} imágenes · "
                  f"{time.perf_counter()-t0:.1f} s", flush=True)
    torch.cuda.synchronize()
    ev,_ = official_modules()
    # Evaluate the exact rounded artifact that will later be submitted.
    return ev.leer_envio(str(destination)).preds,time.perf_counter()-t0
