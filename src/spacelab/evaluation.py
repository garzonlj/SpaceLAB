"""Competition metrics delegated exclusively to the untouched official module."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import read_labels
from .io import digest, write_json


def official_modules():
    kit = Path(__file__).resolve().parents[2]/"kit_participantes"
    sys.path.insert(0,str(kit))
    return importlib.import_module("evaluar"),importlib.import_module("formato_envio")


def score_predictions(root, records, preds, *, threshold=None, tune=True):
    ev,_ = official_modules()
    gt = {r["id"]:read_labels(Path(root)/r["label"]) for r in records}
    if not gt or any(r["image"].startswith("images/oculto/") for r in records):
        raise ValueError("Prohibida evaluación supervisada del oculto")
    if tune and any(r["image"].startswith("images/test/") for r in records):
        raise ValueError("Prohibido ajustar el umbral en test")
    env = ev.Envio(preds=preds)
    pc = ev.preparar(gt,env)
    if tune:
        threshold,_ = ev.barrer_umbral(pc)
    if threshold is None:
        raise ValueError("Evaluación final necesita umbral previamente congelado")
    result = ev.evaluar(gt,env,umbral=float(threshold))
    return result,gt,pc


def diagnostics(gt, preds, threshold):
    """Class-agnostic geometric greedy matching for confusion and FP/FN diagnostics.

    This diagnostic matching is distinct from official class-aware metric matching.
    Rows=true classes, columns=predicted classes, index 5=background.
    """
    ev,_ = official_modules()
    confusion = np.zeros((6,6),int)
    errors = []
    for image_id,g in gt.items():
        p = preds[image_id]
        p = p[p[:,5]>=threshold]
        used = set()
        ious = ev.iou_matriz(p[:,1:5],g[:,1:5])
        for index in np.argsort(-p[:,5],kind="stable"):
            available = [j for j in range(len(g)) if j not in used and ious[index,j]>=.5]
            if available:
                j = max(available,key=lambda j:ious[index,j])
                used.add(j)
                confusion[int(g[j,0]),int(p[index,0])] += 1
                if g[j,0] != p[index,0]:
                    errors.append({"id":image_id,"kind":"class_error","gt":j,"pred":int(index)})
            else:
                confusion[5,int(p[index,0])] += 1
                errors.append({"id":image_id,"kind":"FP","gt":None,"pred":int(index)})
        for j in range(len(g)):
            if j not in used:
                confusion[int(g[j,0]),5] += 1
                errors.append({"id":image_id,"kind":"FN","gt":j,"pred":None})
    return confusion,errors


def sliced_metrics(root, records, gt, preds, threshold, pc):
    ev,_ = official_modules()
    rows = []
    for name,predicate in [("density_0_10",lambda n:n<=10),("density_11_50",lambda n:10<n<=50),
                           ("density_gt50",lambda n:n>50)]:
        ids = {k for k,v in gt.items() if predicate(len(v))}
        if ids:
            metrics = ev.metricas(pc,threshold,ids)
            rows.append({"slice":name,"images":len(ids),**{k:v for k,v in metrics.items() if k!="por_clase"}})
    # Object-size AP/recall: full official matching, then ignore detections matched
    # to out-of-range GT. Unmatched FPs count if their own predicted area is in range.
    # No rematching to smaller objects; this is a custom diagnostic, not COCOeval.
    sized = {"small":(0,32**2),"medium":(32**2,96**2),"large":(96**2,np.inf)}
    for label,(lo,hi) in sized.items():
        per_class = []
        for c in range(5):
            all_conf,all_tp,n_gt = [],[],0
            for row in records:
                g,p = gt[row["id"]],preds[row["id"]]
                g = g[g[:,0]==c]
                p = p[p[:,0]==c]
                p = p[np.argsort(-p[:,5],kind="stable")]
                area = row["width"]*row["height"]
                in_range = (g[:,3]*g[:,4]*area>=lo)&(g[:,3]*g[:,4]*area<hi)
                n_gt += int(in_range.sum())
                ious = ev.iou_matriz(p[:,1:5],g[:,1:5])
                tp = np.zeros((10,len(p)),bool)
                ignore = np.zeros_like(tp)
                for t,cutoff in enumerate(ev.UMBRALES_IOU):
                    used = set()
                    for i in range(len(p)):
                        valid = [j for j in range(len(g)) if j not in used and ious[i,j]>=cutoff-1e-12]
                        if valid:
                            j = max(valid,key=lambda j:ious[i,j])
                            used.add(j)
                            tp[t,i] = in_range[j]
                            ignore[t,i] = not in_range[j]
                        else:
                            predicted_area = p[i,3]*p[i,4]*area
                            ignore[t,i] = not lo<=predicted_area<hi
                all_conf.extend(p[:,5])
                all_tp.append((tp,ignore))
            if n_gt:
                conf = np.array(all_conf)
                tp = np.concatenate([x[0] for x in all_tp],axis=1)
                ignore = np.concatenate([x[1] for x in all_tp],axis=1)
                order = np.argsort(-conf,kind="stable")
                aps = [float(ev._ap(tp[t:t+1,order][:,~ignore[t,order]],n_gt)[0]) for t in range(10)]
                selected = (conf>=threshold)&~ignore[0]
                ntp = int(tp[0,selected].sum())
                per_class.append({"class_id":c,"ap":float(np.mean(aps)),"recall":ntp/n_gt,"n_gt":n_gt})
        if per_class:
            rows.append({"slice":"native_area_"+label,"images":len(records),
                "map50_95":float(np.mean([r["ap"] for r in per_class])),
                "recall_macro":float(np.mean([r["recall"] for r in per_class]))})
    return rows


def save_evaluation(folder, root, records, preds, result, gt, pc, runtime):
    from .visualizations import evaluation_figures
    folder = Path(folder)
    metrics = folder/"metrics"
    metrics.mkdir(parents=True,exist_ok=True)
    ev,_ = official_modules()
    write_json(metrics/"official_score.json",result)
    write_json(metrics/"validation_metrics.json",result["total"])
    pd.DataFrame(result["total"]["por_clase"]).T.to_csv(metrics/"metrics_by_class.csv")
    write_json(metrics/"runtime.json",runtime)
    confusion,errors = diagnostics(gt,preds,result["umbral"])
    names = [*ev.CLASES,"background"]
    pd.DataFrame(confusion,index=names,columns=names).to_csv(metrics/"confusion_matrix.csv")
    pd.DataFrame(errors,columns=["id","kind","gt","pred"]).to_csv(metrics/"errors.csv",index=False)
    pd.DataFrame(sliced_metrics(root,records,gt,preds,result["umbral"],pc)).to_csv(metrics/"slices.csv",index=False)
    write_json(metrics/"evaluation_provenance.json",{"official_evaluator_sha256":digest(Path(ev.__file__)),
        "threshold_tuned_on_internal_val":not any(r["image"].startswith("images/test/") for r in records),
        "confusion_matching":"diagnostic class-agnostic IoU>=.5", "size_slices":"custom native-pixel area; not COCOeval"})
    evaluation_figures(folder,root,records,gt,preds,pc,confusion,result["umbral"])
