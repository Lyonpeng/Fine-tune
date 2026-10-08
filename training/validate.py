"""Validate a trained model on COCO-annotated images.

Runs the model over the selected images of a dataset, then scores the
predictions against the dataset's annotation with pycocotools COCOeval
(the same checks as ModelTraining/coco_evaluation.py), for both boxes and
masks. All metrics are written to <output-dir>/metrics.json.
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.modeling import build_model

from train_net import find_annotation

# Names of COCOeval.stats, in order
STAT_NAMES = [
    "AP", "AP50", "AP75", "APs", "APm", "APl",
    "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"
]


def prepare_ground_truth(annotation_path, image_names, dataset_path, output_dir):
    """Write the annotation restricted to the selected images, with the
    'info' and 'licenses' fields pycocotools expects."""
    with open(annotation_path, "r", encoding="utf-8") as file:
        coco_json = json.load(file)

    if image_names:
        wanted = set(image_names)
        images = [image for image in coco_json["images"] if image["file_name"] in wanted]
        missing = wanted - {image["file_name"] for image in images}
        if missing:
            raise ValueError(
                f"Images not found in the annotation: {', '.join(sorted(missing))}"
            )
        image_ids = {image["id"] for image in images}
        coco_json["images"] = images
        coco_json["annotations"] = [
            annotation for annotation in coco_json["annotations"]
            if annotation["image_id"] in image_ids
        ]

    if not coco_json["images"]:
        raise ValueError("No validation images selected")

    # Annotation names may differ in case from the files (2.JPG vs 2.jpg),
    # which only works on case-insensitive file systems; use the real names
    on_disk = {name.lower(): name for name in os.listdir(dataset_path)}
    for image in coco_json["images"]:
        actual = on_disk.get(image["file_name"].lower())
        if actual is None:
            raise FileNotFoundError(f"Image not found in the dataset folder: {image['file_name']}")
        image["file_name"] = actual

    coco_json.setdefault("info", {})
    coco_json.setdefault("licenses", [])

    ground_truth_path = os.path.join(output_dir, "ground_truth.json")
    with open(ground_truth_path, "w", encoding="utf-8") as file:
        json.dump(coco_json, file)

    return ground_truth_path, coco_json


def setup(args, dataset_name):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = os.path.abspath(args.weights)
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.OUTPUT_DIR = os.path.abspath(args.output_dir)

    if cfg.MODEL.DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available, validating on CPU")
        cfg.MODEL.DEVICE = "cpu"

    cfg.freeze()
    return cfg


def predict(cfg, dataset_name):
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    loader = build_detection_test_loader(cfg, dataset_name)

    # detectron2 numbers categories from 0; map back to the annotation's ids
    # (the mapping is only filled in once the loader has read the dataset)
    metadata = MetadataCatalog.get(dataset_name)
    to_dataset_id = {
        contiguous: dataset_id
        for dataset_id, contiguous in metadata.thing_dataset_id_to_contiguous_id.items()
    }
    total = len(loader)
    predictions = []

    with torch.no_grad():
        for index, inputs in enumerate(loader, start=1):
            outputs = model(inputs)
            for item, output in zip(inputs, outputs):
                instances = output["instances"].to("cpu")
                for prediction in instances_to_coco_json(instances, item["image_id"]):
                    prediction["category_id"] = to_dataset_id[prediction["category_id"]]
                    predictions.append(prediction)
            # read by the web app to show progress
            print(f"PROGRESS {index}/{total}", flush=True)

    return predictions


def evaluate(coco_gt, coco_dt, iou_type):
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()

    summary = io.StringIO()
    with contextlib.redirect_stdout(summary):
        coco_eval.summarize()
    print(f"\n[{iou_type}]\n{summary.getvalue()}")

    iou_index = np.where(np.isclose(coco_eval.params.iouThrs, 0.50))[0][0]

    # precision: [IoU, Recall, Category, Area, MaxDets], area "all", maxDets 100
    precision_50 = coco_eval.eval["precision"][iou_index, :, :, 0, 2]
    precision_50 = precision_50[precision_50 > -1]
    precision = float(np.mean(precision_50)) if precision_50.size else 0.0

    # recall: [IoU, Category, Area, MaxDets]
    recall_50 = coco_eval.eval["recall"][iou_index, :, 0, 2]
    recall_50 = recall_50[recall_50 > -1]
    recall = float(np.mean(recall_50)) if recall_50.size else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "stats": dict(zip(STAT_NAMES, (float(value) for value in coco_eval.stats))),
        "precision50": precision,
        "recall50": recall,
        "f1_50": f1,
        "summary": summary.getvalue()
    }


def main(args):
    dataset_path = os.path.abspath(args.dataset)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    os.makedirs(args.output_dir, exist_ok=True)

    image_names = []
    if args.images_file:
        with open(args.images_file, "r", encoding="utf-8") as file:
            image_names = json.load(file)

    annotation_path = find_annotation(dataset_path)
    ground_truth_path, ground_truth = prepare_ground_truth(
        annotation_path, image_names, dataset_path, args.output_dir
    )

    dataset_name = f"validate_{os.path.basename(os.path.normpath(args.output_dir))}"
    register_coco_instances(dataset_name, {}, ground_truth_path, dataset_path)

    cfg = setup(args, dataset_name)
    print(f"Dataset: {dataset_path}")
    print(f"Annotation: {annotation_path}")
    print(f"Images: {len(ground_truth['images'])}")
    print(f"Weights: {cfg.MODEL.WEIGHTS}")
    print(f"Score threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")

    predictions = predict(cfg, dataset_name)
    with open(os.path.join(args.output_dir, "coco_instances_results.json"), "w", encoding="utf-8") as file:
        json.dump(predictions, file)

    iou_types = ["bbox", "segm"] if cfg.MODEL.MASK_ON else ["bbox"]
    metrics = {
        "images": len(ground_truth["images"]),
        "annotations": len(ground_truth["annotations"]),
        "predictions": len(predictions),
        "scoreThreshold": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        "results": {}
    }

    if predictions:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt = COCO(ground_truth_path)
            coco_dt = coco_gt.loadRes(predictions)
        for iou_type in iou_types:
            metrics["results"][iou_type] = evaluate(coco_gt, coco_dt, iou_type)
    else:
        # loadRes cannot take an empty list; nothing detected scores zero
        print("The model made no predictions on the selected images")
        for iou_type in iou_types:
            metrics["results"][iou_type] = {
                "stats": dict.fromkeys(STAT_NAMES, 0.0),
                "precision50": 0.0, "recall50": 0.0, "f1_50": 0.0, "summary": ""
            }

    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(f"Metrics saved: {os.path.join(args.output_dir, 'metrics.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        required=True,
        help="config.yaml of the training run that produced the model"
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the trained model (.pth)"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Full path to the validation dataset folder"
    )

    parser.add_argument(
        "--images-file",
        help="JSON list of image file names to validate on (default: all annotated images)"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for predictions and metrics.json"
    )

    main(parser.parse_args())
