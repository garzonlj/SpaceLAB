"""Explicit PyTorch adapters; native Ultralytics automatic trainer is NOT used.

This avoids hidden augmentations and lets both YOLO architectures share an exact
loop. Internal Ultralytics APIs are deliberately isolated here and need a Colab
compatibility check before relying on experimental outcomes.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from .io import CLASSES


class Detector:
    def __init__(self, config, pretrained=True):
        self.config = config
        self.family = "frcnn" if config["model"] == "faster_rcnn_mobilenet" else "yolo"
        if self.family == "yolo":
            from ultralytics import YOLO
            from ultralytics.cfg import get_cfg
            from ultralytics.nn.tasks import DetectionModel
            # Ultralytics load() rebuilds classification branches at the source
            # width. Apply that same construction before restoring our complete
            # state dict, otherwise nc=5 builds a different hidden head width.
            source = YOLO(config["model"] + ".pt")
            self.net = DetectionModel(deepcopy(source.model.yaml), ch=3, nc=5, verbose=False)
            self.net.load(source.model, verbose=pretrained)
            self.net.args = get_cfg()
            self.net.args.box = config["loss_box"]
            self.net.args.cls = config["loss_cls"]
            self.net.args.dfl = config["loss_dfl"]
            self.net.args.epochs = config["epochs"]
            self.net.names = dict(enumerate(CLASSES))
            self.net.to("cpu")
            del source
            self.backbone_prefixes = tuple(f"model.{i}." for i in range(len(self.net.yaml["backbone"])))
            self.end2end = bool(getattr(self.net,"end2end",False))
            # Keep native end-to-end mode; no claim that NMS changes YOLO26 output.
            self.net.model[-1].max_det = 300
        else:
            from torchvision.models.detection import (fasterrcnn_mobilenet_v3_large_fpn,
                FasterRCNN_MobileNet_V3_Large_FPN_Weights)
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            self.net = fasterrcnn_mobilenet_v3_large_fpn(
                weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None,
                weights_backbone=None, min_size=config["imgsz"], max_size=config["imgsz"],
                box_score_thresh=config["confidence_floor"],box_detections_per_img=300,
                box_nms_thresh=config["nms_iou"],rpn_pre_nms_top_n_test=4000,rpn_post_nms_top_n_test=2000)
            self.net.roi_heads.box_predictor = FastRCNNPredictor(self.net.roi_heads.box_predictor.cls_score.in_features,6)
            self.backbone_prefixes = ("backbone.body.",)
            self.end2end = False

    def to(self, device):
        self.net.to(device)
        return self

    def policy(self, mode, epoch):
        frozen = mode == "partial" or (mode == "progressive" and epoch < self.config["unfreeze_epoch"])
        selected = []
        for name,p in self.net.named_parameters():
            is_backbone = name.startswith(self.backbone_prefixes)
            # DFL projection in YOLO is a fixed discretization layer, not a trainable backbone.
            fixed = ".dfl." in name
            p.requires_grad_(not fixed and not (frozen and is_backbone))
            if not p.requires_grad:
                selected.append(name)
        return {"epoch":epoch,"frozen_parameter_names":selected,
            "trainable_parameters":sum(p.numel() for p in self.net.parameters() if p.requires_grad),
            "backbone_prefixes":list(self.backbone_prefixes)}

    def train_mode(self):
        self.net.train()
        for name,module in self.net.named_modules():
            if name.startswith(self.backbone_prefixes) and not any(p.requires_grad for p in module.parameters()):
                module.eval()

    def loss_epoch(self, epoch):
        """Reproduce the native end-to-end loss schedule for the current epoch."""
        if self.family != "yolo":
            return {}
        if getattr(self.net, "criterion", None) is None:
            self.net.criterion = self.net.init_criterion()
        criterion = self.net.criterion
        if all(hasattr(criterion, key) for key in ("decay", "o2m", "o2o", "total", "updates")):
            criterion.updates = epoch
            criterion.o2m = criterion.decay(epoch)
            criterion.o2o = max(criterion.total-criterion.o2m, 0)
            return {"one_to_many":criterion.o2m,"one_to_one":criterion.o2o}
        return {}

    def loss(self, images, labels, validation=False):
        if self.family == "yolo":
            count = len(images)
            classes = torch.cat([y[:,0:1] for y in labels],dim=0)
            boxes = torch.cat([y[:,1:5] for y in labels],dim=0)
            indices = torch.cat([torch.full((len(y),),i,device=images.device) for i,y in enumerate(labels)])
            result, items = self.net.loss({"img":images,"cls":classes,"bboxes":boxes,"batch_idx":indices})
            if isinstance(items, dict):
                components = {str(k):float(v.detach().sum()) for k,v in items.items()}
            else:
                components = {f"component_{i}":float(v) for i,v in enumerate(items.detach().flatten())}
            return result.sum()/count, components
        targets = []
        size = images.shape[-1]
        for y in labels:
            xyxy = torch.cat([y[:,1:3]-y[:,3:5]/2,y[:,1:3]+y[:,3:5]/2],dim=1)*size
            targets.append({"boxes":xyxy,"labels":y[:,0].long()+1})
        # Torchvision exposes validation losses only in train mode. Freeze BN and
        # dropout to avoid mutating statistics; proposal sampling remains stochastic.
        if validation:
            self.net.train()
            for module in self.net.modules():
                if isinstance(module,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout)):
                    module.eval()
        result = self.net(list(images),targets)
        return sum(result.values()), {k:float(v.detach()) for k,v in result.items()}

    @torch.no_grad()
    def predict(self, images, confidence=None, nms_iou=None):
        self.net.eval()
        confidence = self.config["confidence_floor"] if confidence is None else confidence
        nms_iou = self.config["nms_iou"] if nms_iou is None else nms_iou
        if self.family == "yolo":
            try:
                from ultralytics.utils.nms import non_max_suppression
            except ImportError:
                from ultralytics.utils.ops import non_max_suppression
            raw = self.net(images)
            result = non_max_suppression(raw,conf_thres=confidence,iou_thres=nms_iou,
                max_det=300,nc=5,end2end=self.end2end,max_nms=100000,max_time_img=10.)
            return [r[:,[5,0,1,2,3,4]].float().cpu().numpy() for r in result]
        self.net.roi_heads.score_thresh = confidence
        self.net.roi_heads.nms_thresh = nms_iou
        result = self.net(list(images))
        return [torch.cat([(r["labels"]-1).float()[:,None],r["boxes"],r["scores"][:,None]],dim=1).cpu().numpy() for r in result]
