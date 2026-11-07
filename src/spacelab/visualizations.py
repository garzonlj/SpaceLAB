"""Figures from persisted real predictions/history only; no placeholder values."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from .io import CLASSES


def _save(folder,name):
    folder = Path(folder)/"figures"
    folder.mkdir(parents=True,exist_ok=True)
    plt.tight_layout()
    plt.savefig(folder/name,bbox_inches="tight",dpi=140)
    plt.close()


def learning_curves(folder):
    path = Path(folder)/"metrics/training_history.csv"
    if not path.exists():
        return
    h = pd.read_csv(path)
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    for key in ["train_loss","val_loss"]:
        axes[0].plot(h.epoch,h[key],label=key)
    for key in ["score","map50_95","f1_macro"]:
        axes[1].plot(h.epoch,h[key],label=key)
    for name in CLASSES:
        if name+"_ap" in h:
            axes[2].plot(h.epoch,h[name+"_ap"],label=name)
    for ax in axes:
        ax.set_xlabel("Época")
        ax.legend(fontsize=7)
    axes[0].set_title("Pérdidas: condiciones train/val diferentes")
    axes[1].set_title("Validación interna; máximo oficial para checkpoint")
    axes[2].set_title("AP por clase")
    _save(folder,"learning_curves.png")


def evaluation_figures(folder,root,records,gt,preds,pc,confusion,threshold):
    fig,ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(confusion,cmap="Blues")
    ax.set_xticks(range(6),CLASSES+["background"],rotation=45,ha="right")
    ax.set_yticks(range(6),CLASSES+["background"])
    ax.set(xlabel="Predicción",ylabel="Ground truth",title="Confusión diagnóstica IoU≥0.5")
    fig.colorbar(im,ax=ax)
    _save(folder,"confusion_matrix.png")
    fig,ax = plt.subplots(figsize=(7,5))
    for name,k in zip(CLASSES,pc):
        n_gt = sum(k.n_gt_img.values())
        if not n_gt or not len(k.conf):
            continue
        order = np.argsort(-k.conf,kind="stable")
        tp = np.cumsum(k.tp[0,order])
        ax.plot(tp/n_gt,tp/np.arange(1,len(tp)+1),label=name)
    ax.set(xlabel="Recall",ylabel="Precision",title="PR a IoU=0.5; todas las confidencias guardadas")
    ax.legend()
    _save(folder,"precision_recall.png")
    fig,axes = plt.subplots(1,3,figsize=(15,5))
    for ax,row in zip(axes,records[:3]):
        with Image.open(Path(root)/row["image"]) as image:
            w,h = image.size
            ax.imshow(image)
        for values,color in [(gt[row["id"]],"lime"),(preds[row["id"]],"red")]:
            for v in values:
                if len(v)==6 and v[5]<threshold:
                    continue
                ax.add_patch(Rectangle(((v[1]-v[3]/2)*w,(v[2]-v[4]/2)*h),v[3]*w,v[4]*h,
                                       fill=False,color=color,linewidth=.7))
        ax.set_title(row["id"],fontsize=7)
        ax.axis("off")
    _save(folder,"predictions_examples.png")
    learning_curves(folder)


def comparison_figures(table,folder):
    if table.empty:
        print("Sin resultados comparables completos: no se generan rankings ni conclusiones.")
        return
    keys = [k for k in ["score","map50_95","f1_macro","train_wall_seconds","peak_vram_bytes"] if k in table]
    fig,axes = plt.subplots(1,len(keys),figsize=(5*len(keys),4),squeeze=False)
    for ax,key in zip(axes.flat,keys):
        ax.barh(table["run_id"],table[key])
        ax.set_title(key)
    _save(folder,"comparison.png")
