import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spacelab.evaluation import score_predictions
from spacelab.inference import predict_image, undo_flip_boxes
from spacelab.models import DetectorEnsemble
from spacelab.experiment_registry import freeze_selection
from spacelab.io import write_json
from spacelab.time_budget import Deadline


class FakeDetector:
    def __init__(self, outputs):
        self.net = torch.nn.Linear(1, 1)
        self.outputs = outputs
        self.index = 0

    def predict(self, images, confidence=None, nms_iou=None):
        result = self.outputs[min(self.index,len(self.outputs)-1)].copy()
        self.index += 1
        return [result]


class InferenceRefinementTests(unittest.TestCase):
    def test_probe_scores_cannot_be_frozen_as_final_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root/"experiment_manifest.json",{"status":"completado","config":{"model":"yolo26n"},"protocol_id":"fixture"})
            (root/"checkpoints").mkdir()
            (root/"checkpoints/best.pt").write_bytes(b"fixture")
            write_json(root/"metrics/official_score.json",{"evaluation_split":"internal_val_probe"})
            with self.assertRaisesRegex(ValueError,"validación interna"):
                freeze_selection(root,{"protocol_id":"fixture"},root/"frozen.json")

    def test_flip_round_trip_preserves_class_and_score(self):
        boxes = np.array([[2,10,5,40,30,0.7]], dtype=np.float32)
        for mode in ("hflip","vflip"):
            result = undo_flip_boxes(undo_flip_boxes(boxes,100,50,mode),100,50,mode)
            np.testing.assert_allclose(result,boxes)

    def test_tta_undoes_letterbox_and_flip_before_merging(self):
        original = np.array([[0,10,30,40,55,0.8]], dtype=np.float32)
        flipped = np.array([[0,60,30,90,55,0.8]], dtype=np.float32)
        detector = FakeDetector([original,flipped])
        config = {"imgsz":100,"amp":False,"nms_iou":0.6,"tta":True}
        result = predict_image(detector,Image.new("RGB",(100,50)),config,Deadline(float("inf")))
        np.testing.assert_allclose(result,[[0,10,5,40,30,0.8]],rtol=1e-6)

    def test_ensemble_nms_suppresses_duplicates_within_class(self):
        a = np.array([[0,10,30,40,55,0.9]],dtype=np.float32)
        b = np.array([[0,10,30,40,55,0.8],[1,10,30,40,55,0.7]],dtype=np.float32)
        predictor = DetectorEnsemble([FakeDetector([a]),FakeDetector([b])])
        config = {"imgsz":100,"amp":False,"nms_iou":0.6,"ensemble_run":"fixture"}
        result = predict_image(predictor,Image.new("RGB",(100,50)),config,Deadline(float("inf")))
        self.assertEqual(len(result),2)
        self.assertEqual(set(result[:,0]),{0,1})

    def test_low_threshold_is_representable_and_improves_official_f1(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"label.txt").write_text("0 0.5 0.5 0.2 0.2\n",encoding="utf-8")
            records = [{"id":"fixture","image":"images/train/fixture.jpg","label":"label.txt"}]
            preds = {"fixture":np.array([[0,0.5,0.5,0.2,0.2,0.02],[0,0.1,0.1,0.1,0.1,0.001]])}
            baseline,_,_ = score_predictions(root,records,preds)
            tuned,_,_ = score_predictions(root,records,preds,thresholds=[round(i/100,2) for i in range(91)])
            self.assertGreater(tuned["total"]["f1_macro"],baseline["total"]["f1_macro"])
            self.assertEqual(tuned["umbral"],0.01)
            self.assertEqual(float(f"{tuned['umbral']:.2f}"),tuned["umbral"])


if __name__ == "__main__":
    unittest.main()
