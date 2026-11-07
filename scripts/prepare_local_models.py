"""Download the four declared pretrained backbones into local ignored caches."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".eda_tmp/ultralytics"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "artifacts/torch"))

if __name__ == "__main__":
    import torch
    from ultralytics import YOLO
    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights

    torch.set_num_threads(4)
    os.chdir(ROOT)
    for name in ("yolo26n", "yolo11n", "yolo26s"):
        print("Preparando pesos:", name, flush=True)
        model = YOLO(name + ".pt")
        del model
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    weights.get_state_dict(progress=True, check_hash=True)
    print("Pesos locales preparados.", flush=True)
