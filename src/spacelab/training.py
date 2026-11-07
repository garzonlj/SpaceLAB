"""Shared controlled training, measured calibration and epoch-level recovery."""
import gc
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .checkpointing import initialize_weights, restore, save_checkpoint
from .dataset import DetectionDataset, collate, oversample
from .evaluation import score_predictions, save_evaluation
from .inference import predictions
from .io import fingerprint, read_json, write_json
from .models import Detector
from .splits import select_records, describe
from .time_budget import DeadlineExceeded


def environment():
    from importlib.metadata import version
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere una GPU CUDA y una instalación de PyTorch con CUDA.")
    props = torch.cuda.get_device_properties(0)
    free,total = torch.cuda.mem_get_info()
    return {"gpu":props.name,"vram_bytes":total,"free_bytes":free,"cuda":torch.version.cuda,
        "libraries":{p:version(p) for p in ["torch","torchvision","ultralytics","numpy","pandas","Pillow"]}}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Detection CUDA kernels may not be bitwise deterministic; record the environment.


def make_optimizer(detector, config):
    # All non-DFL parameters remain in the optimizer from the start; frozen ones
    # have no gradient/state. Unfreezing activates them without dropping momentum
    # of already trained modules. No arbitrary same layer index across architectures.
    backbone,head = [],[]
    for name,p in detector.net.named_parameters():
        if ".dfl." in name:
            continue
        (backbone if name.startswith(detector.backbone_prefixes) else head).append(p)
    groups = [{"params":backbone,"name":"backbone"},{"params":head,"name":"neck_head"}]
    if config["optimizer"] == "SGD":
        return torch.optim.SGD(groups,lr=config["lr"],momentum=config["momentum"],weight_decay=config["weight_decay"])
    if config["optimizer"] != "AdamW":
        raise ValueError("Optimizador no soportado")
    return torch.optim.AdamW(groups,lr=config["lr"],betas=tuple(config["betas"]),weight_decay=config["weight_decay"])


def loader(root, records, config, training, epoch=0, batch=None):
    ds = DetectionDataset(root,records,config,training,epoch)
    return DataLoader(ds,batch_size=batch or config["batch"],shuffle=training,
        generator=torch.Generator().manual_seed(config["seed"]+epoch),collate_fn=collate,
        num_workers=config["workers"],pin_memory=True,drop_last=False)


def epoch_pass(detector, dl, config, deadline, optimizer=None, scaler=None, max_batches=None):
    training = optimizer is not None
    if training:
        detector.train_mode()
    else:
        detector.net.eval()
    total_loss,n,read_seconds,batch_times = 0.,0,0.,[]
    prepare_seconds,verify_seconds = 0.,0.
    iterator = iter(dl)
    while True:
        deadline.check(15)
        t0 = time.perf_counter()
        try:
            images,labels,reads = next(iterator)
        except StopIteration:
            break
        if max_batches is not None and len(batch_times)>=max_batches:
            break
        images = images.cuda(non_blocking=True)
        labels = [y.cuda(non_blocking=True) for y in labels]
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast("cuda",dtype=torch.float16,enabled=config["amp"]):
            loss,components = detector.loss(images,labels,validation=not training)
        if not torch.isfinite(loss):
            raise RuntimeError("Pérdida no finita: experimento interrumpido, no completar ni promover")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.net.parameters(),10.)
            scaler.step(optimizer)
            scaler.update()
        torch.cuda.synchronize()
        batch_times.append(time.perf_counter()-t0)
        total_loss += float(loss.detach())*len(images)
        n += len(images)
        if len(batch_times) % 50 == 0:
            print(f"{'train' if training else 'val_loss'}: {len(batch_times)}/{len(dl)} batches; "
                  f"loss={total_loss/n:.5f}", flush=True)
        read_seconds += reads["image_read_seconds"]
        prepare_seconds += reads["total_prepare_seconds"]
        verify_seconds += reads["verify_seconds"]
    if not n:
        raise ValueError("No se procesó ningún batch")
    return {"loss":total_loss/n,"images":n,"batches":len(batch_times),"read_seconds":read_seconds,
            "prepare_seconds":prepare_seconds,"verify_seconds":verify_seconds,
            "batch_seconds_mean":float(np.mean(batch_times)),"batch_seconds":batch_times}


def calibration_key(config):
    return fingerprint({k:config[k] for k in ["model","imgsz","batch","amp","augmentation","freeze","optimizer"]})


def benchmark(config, root, protocol, deadline, folder):
    env = environment()
    seed_all(config["seed"])
    folder = Path(folder)
    device_model = None
    try:
        device_model = Detector(config).to("cuda")
        device_model.policy(config["freeze"],0)
        optimizer = make_optimizer(device_model,config)
        scaler = torch.amp.GradScaler("cuda",enabled=config["amp"])
        records = select_records(protocol,"screening")
        torch.cuda.reset_peak_memory_stats()
        # Warm-up is timed separately and counts against the shared budget.
        warm = epoch_pass(device_model,loader(root,records,config,True),config,deadline,optimizer,scaler,max_batches=1)
        measured = epoch_pass(device_model,loader(root,records,config,True),config,deadline,optimizer,scaler,
                               max_batches=config.get("benchmark_batches",4))
        dense_records = sorted(select_records(protocol,"train"),key=lambda r:-sum(r["counts"]))[:config["batch"]]
        stress = epoch_pass(device_model,loader(root,dense_records,config,True),config,deadline,optimizer,scaler,max_batches=1)
        sample = select_records(protocol,"val")[:8]
        _,seconds = predictions(device_model,root,sample,config,deadline,folder/"benchmark_predictions")
        seconds_per_epoch = measured["batch_seconds_mean"]*math.ceil(len(records)/config["batch"])
        result = {"key":calibration_key(config),"config":config,"environment":env,"warmup":warm,
            "measured":measured,"validation_seconds_per_image":seconds/len(sample),
            "dense_batch_stress":stress,"dense_batch_ids":[r["id"] for r in dense_records],
            "screening_epoch_projection_seconds":seconds_per_epoch,
            "requested_epoch_projection_seconds":seconds_per_epoch*config["epochs"],
            "peak_vram_bytes":torch.cuda.max_memory_allocated(),"measured_at":time.time(),
            "warning":"Muestra corta y stress de densidad de train interno. Proyección aproximada, no época completa medida."}
        write_json(folder/"calibration.json",result)
        return result
    finally:
        del device_model
        gc.collect()
        torch.cuda.empty_cache()


def estimate(config, protocol, calibration):
    count = len(select_records(protocol, "screening" if config["phase"]=="screening" else "train"))
    count += sum(config["oversample_factor"]-1 for r in select_records(protocol,
        "screening" if config["phase"]=="screening" else "train") if r["counts"][4])
    train_epoch = calibration["measured"]["batch_seconds_mean"]*math.ceil(count/config["batch"])
    val_count = len(protocol["val_ids"])
    val_epoch = val_count*calibration["validation_seconds_per_image"]
    # Validation-loss pass is extra, not included in inference throughput; bound
    # its initial projection using measured training batch cost until real epochs exist.
    val_loss = calibration["measured"]["batch_seconds_mean"]*math.ceil(val_count/config["batch"])
    return 1.3*config["epochs"]*(train_epoch+val_epoch+val_loss+10)+30


def train(config, root, protocol, deadline, folder, resume=None):
    from .visualizations import learning_curves
    folder = Path(folder)
    metadata = environment()
    seed_all(config["seed"])
    train_records = select_records(protocol,"screening" if config["phase"]=="screening" else "train")
    expanded = oversample(train_records,config["oversample_factor"])
    write_json(folder/"sampling_distribution.json",{"original":describe(train_records),"effective":describe(expanded)})
    val_records = select_records(protocol,"val")
    detector = Detector(config,pretrained=not bool(resume)).to("cuda")
    if not resume and config.get("initial_checkpoint"):
        initialize_weights(config["initial_checkpoint"],detector,config,protocol["protocol_id"])
        print("Continuando pesos del checkpoint validado:", config["initial_checkpoint"], flush=True)
    optimizer = make_optimizer(detector,config)
    scaler = torch.amp.GradScaler("cuda",enabled=config["amp"])
    history,start,best,bad_epochs = [],0,-1.,0
    if resume:
        state = restore(resume,detector,config,protocol["protocol_id"],optimizer,scaler)
        history,start,best = state["history"],state["epoch"],state["best_score"]
        if history:
            best_epoch = max(history,key=lambda r:r["score"])["epoch"]
            bad_epochs = max(0,start-best_epoch)
    t0 = time.perf_counter()
    state_name = "interrumpido"
    freeze_log,oom_log = [],[]
    effective_batch = config["batch"]
    torch.cuda.reset_peak_memory_stats()
    # Baseline rollback checkpoint makes the first epoch recoverable too.
    save_checkpoint(folder/"checkpoints/last.pt",detector,optimizer,scaler,start,best,config,protocol["protocol_id"],history)
    try:
        for epoch in range(start,config["epochs"]):
            print(f"Época {epoch+1}/{config['epochs']} · {config['id']} · {config['model']}", flush=True)
            deadline.check(30)
            if history:
                recent = history[-1]["epoch_seconds"]
                if deadline.remaining() < 1.2*recent+20:
                    raise DeadlineExceeded("La última época medida no cabe junto con validación y guardado")
            freeze_log.append(detector.policy(config["freeze"],epoch))
            if config.get("loss_schedule") == "native_epoch":
                freeze_log[-1]["loss_schedule"] = detector.loss_epoch(epoch)
            write_json(folder/"metrics/freeze_policy.json",freeze_log)
            tick = time.perf_counter()
            while True:
                try:
                    training = epoch_pass(detector,loader(root,expanded,config,True,epoch,effective_batch),
                                           config,deadline,optimizer,scaler)
                    validation = epoch_pass(detector,loader(root,val_records,config,False,batch=effective_batch),
                                             config,deadline)
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    if effective_batch == 1:
                        raise RuntimeError("OOM en batch=1; no se cambia resolución silenciosamente")
                    effective_batch = max(1,effective_batch//2)
                    oom_log.append({"epoch":epoch,"new_batch":effective_batch})
                    write_json(folder/"metrics/oom_recovery.json",oom_log)
                    # Roll back the complete epoch, optimizer, scaler and RNG.
                    restore(folder/"checkpoints/last.pt",detector,config,protocol["protocol_id"],optimizer,scaler)
                    detector.policy(config["freeze"],epoch)
            preds,infer_seconds = predictions(detector,root,val_records,config,deadline,folder/"predictions/internal_val")
            deadline.check(10)
            result,gt,pc = score_predictions(root,val_records,preds,tune=True)
            score = result["total"]["score"]
            print(f"Época {epoch+1}: score={score:.6f}; train_loss={training['loss']:.5f}; "
                  f"val_loss={validation['loss']:.5f}", flush=True)
            row = {"epoch":epoch+1,"train_loss":training["loss"],"val_loss":validation["loss"],
                "train_seconds":sum(training["batch_seconds"]),"val_loss_seconds":sum(validation["batch_seconds"]),
                "inference_seconds":infer_seconds,"epoch_seconds":time.perf_counter()-tick,
                "effective_batch":effective_batch,"peak_vram_bytes":torch.cuda.max_memory_allocated(),
                "lr":optimizer.param_groups[0]["lr"],"threshold":result["umbral"],
                **{k:v for k,v in result["total"].items() if k!="por_clase"}}
            for name,values in result["total"]["por_clase"].items():
                row[name+"_ap"] = values["ap50_95"]
                row[name+"_f1"] = values["f1"]
            history.append(row)
            metrics = folder/"metrics"
            metrics.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(history).to_csv(metrics/"training_history.csv",index=False)
            improved = score>best
            best = max(best,score)
            bad_epochs = 0 if improved else bad_epochs+1
            save_checkpoint(folder/"checkpoints/last.pt",detector,optimizer,scaler,epoch+1,best,config,protocol["protocol_id"],history)
            if improved:
                save_checkpoint(folder/"checkpoints/best.pt",detector,optimizer,scaler,epoch+1,best,config,protocol["protocol_id"],history)
                write_json(metrics/"best_epoch.json",{"epoch":epoch+1,"score":best,"threshold":result["umbral"]})
            row["checkpoint_seconds"] = time.perf_counter()-tick-row["epoch_seconds"]
            row["epoch_seconds"] = time.perf_counter()-tick
            pd.DataFrame(history).to_csv(metrics/"training_history.csv",index=False)
            if epoch+1 == config["epochs"]:
                state_name = "completado"
            if config["patience"] and bad_epochs>=config["patience"]:
                # Early-stopped runs did not complete equal target epochs.
                state_name = "early_stopped"
                break
        learning_curves(folder)
        return {"status":state_name,"completed_epochs":len(history),"target_epochs":config["epochs"],
                "best_score":best,"effective_batch":effective_batch,"oom_recovery":oom_log,
                "train_wall_seconds":time.perf_counter()-t0,"environment":metadata,
                "peak_vram_bytes":torch.cuda.max_memory_allocated()}
    finally:
        write_json(folder/"metrics/training_runtime.json",{"wall_seconds":time.perf_counter()-t0,
            "completed_epochs":len(history),"peak_vram_bytes":torch.cuda.max_memory_allocated(),"oom_recovery":oom_log})
        del detector,optimizer,scaler
        gc.collect()
        torch.cuda.empty_cache()
