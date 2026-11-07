"""Small atomic metadata helpers; paths are resolved only when called."""
import hashlib
import json
import os
from pathlib import Path

CLASSES = ["small_boat", "cargo_ship", "fishing_vessel", "warship", "work_vessel"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def dataset_root(explicit):
    root = Path(explicit).expanduser().resolve()
    candidates = [root, root / "spacelab_dataset", root / "spacelab_dataset/spacelab_dataset"]
    found = [p for p in candidates if (p / "data.yaml").is_file() and (p / "images/train").is_dir()]
    if len(found) != 1:
        raise ValueError(f"Seleccione una raíz única con data.yaml e images/train. Examinadas: {candidates}")
    return found[0]
