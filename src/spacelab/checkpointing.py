"""Atomic trusted checkpoints with optimizer, scaler, RNG and protocol identity."""
import os
import random
from pathlib import Path

import numpy as np
import torch

from .io import digest


def initialize_weights(path, detector, config, protocol_id):
    """Warm-start a new phase from this project's checkpoint; reset optimizer/RNG."""
    if digest(path) != config["initial_checkpoint_sha256"]:
        raise ValueError("El checkpoint inicial cambió desde la selección.")
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data["protocol_id"] != protocol_id or data["config"]["model"] != config["model"]:
        raise ValueError("El checkpoint inicial pertenece a otro modelo o protocolo.")
    detector.net.load_state_dict(data["model"], strict=True)


def save_checkpoint(path, detector, optimizer, scaler, epoch, score, config, protocol_id, history):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # State dicts only: avoid serializing an architecture-dependent Python model.
    torch.save({"model":{k:v.detach().cpu() for k,v in detector.net.state_dict().items()},
        "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"best_score":score,
        "config":config,"protocol_id":protocol_id,"history":history,
        "rng_python":random.getstate(),"rng_numpy":np.random.get_state(),"rng_torch":torch.get_rng_state(),
        "rng_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},tmp)
    os.replace(tmp,path)


def restore(path, detector, config, protocol_id, optimizer=None, scaler=None):
    # Only restore checkpoints created by this project/team, never untrusted pickle files.
    data = torch.load(path,map_location="cpu",weights_only=False)
    if data["protocol_id"] != protocol_id or data["config"] != config:
        raise ValueError("Checkpoint incompatible: split o configuración diferente")
    detector.net.load_state_dict(data["model"])
    if optimizer is not None:
        optimizer.load_state_dict(data["optimizer"])
        for state in optimizer.state.values():
            for key,value in state.items():
                if isinstance(value,torch.Tensor):
                    state[key] = value.to(next(detector.net.parameters()).device)
        scaler.load_state_dict(data["scaler"])
        random.setstate(data["rng_python"])
        np.random.set_state(data["rng_numpy"])
        torch.set_rng_state(data["rng_torch"])
        if data["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(data["rng_cuda"])
    return data
