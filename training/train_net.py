import argparse
import json
import os
import matplotlib.pyplot as plt

from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_setup


def find_annotation(dataset_path):
    json_files = [
        file for file in os.listdir(dataset_path)
        if file.lower().endswith(".json")
    ]

    if len(json_files) == 0:
        raise FileNotFoundError(
            f"No JSON annotation found in: {dataset_path}"
        )

    if len(json_files) > 1:
        raise ValueError(
            f"Multiple JSON files found in dataset: {json_files}"
        )

    return os.path.join(dataset_path, json_files[0])


def parse_advanced_config(config_text):
    opts = []

    if not config_text:
        return opts

    blocked_keys = {
        "DATASETS.TRAIN",
        "DATASETS.TEST",
        "OUTPUT_DIR"
    }

    for line_number, line in enumerate(config_text.splitlines(), start=1):
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        parts = line.split(maxsplit=1)

        if len(parts) != 2:
            raise ValueError(
                f"Invalid advanced configuration at line {line_number}: {line}"
            )

        key, value = parts

        if key in blocked_keys:
            raise ValueError(
                f"Advanced configuration cannot override {key}"
            )

        opts.extend([key, value])

    return opts


def plot_training_loss(output_dir):
    metrics_path = os.path.join(output_dir, "metrics.json")

    if not os.path.isfile(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return

    iterations = []
    total_losses = []

    with open(metrics_path, "r") as file:
        for line in file:
            try:
                data = json.loads(line)

                if "iteration" in data and "total_loss" in data:
                    iterations.append(data["iteration"])
                    total_losses.append(data["total_loss"])

            except json.JSONDecodeError:
                continue

    if not iterations:
        print("No training loss data found.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, total_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.tight_layout()

    graph_path = os.path.join(
        output_dir,
        "training_loss.png"
    )

    plt.savefig(graph_path, dpi=300)
    plt.close()

    print(f"Loss graph saved: {graph_path}")


def setup(args):
    # --dataset is now the full dataset path
    dataset_path = os.path.abspath(args.dataset)

    # Get folder name from path
    dataset_folder = os.path.basename(
        os.path.normpath(dataset_path)
    )

    # Unique Detectron2 dataset registration name
    dataset_name = f"train_{dataset_folder}"

    cfg = get_cfg()

    # 1. Load base YAML
    cfg.merge_from_file(args.config_file)

    # 2. Web application controlled settings
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()

    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.max_iterations

    if args.weights == "none":
        cfg.MODEL.WEIGHTS = ""

    # 4. Output directory
    cfg.OUTPUT_DIR = os.path.abspath(
        args.output_dir or os.path.join(
            args.output_root,
            dataset_folder
        )
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Save the full configuration even when an override is invalid.
    try:
        advanced_opts = parse_advanced_config(args.advanced_config)
        if advanced_opts:
            cfg.merge_from_list(advanced_opts)
    finally:
        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w", encoding="utf-8") as file:
            file.write(cfg.dump())

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_path}")
    annotation_path = find_annotation(dataset_path)
    register_coco_instances(dataset_name, {}, annotation_path, dataset_path)

    cfg.freeze()

    default_setup(cfg, args)

    print(f"Dataset: {dataset_folder}")
    print(f"Dataset path: {dataset_path}")
    print(f"Annotation: {annotation_path}")
    print(f"Config: {args.config_file}")
    print(f"Transfer learning: {args.weights}")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Training minimum sizes: {cfg.INPUT.MIN_SIZE_TRAIN}")
    print(f"Training maximum size: {cfg.INPUT.MAX_SIZE_TRAIN}")
    print(f"Weights: {cfg.MODEL.WEIGHTS}")
    print(f"Output: {cfg.OUTPUT_DIR}")

    if advanced_opts:
        print("Advanced overrides:")

        for i in range(0, len(advanced_opts), 2):
            print(
                f"  {advanced_opts[i]} = "
                f"{advanced_opts[i + 1]}"
            )

    return cfg


def main(args):
    cfg = setup(args)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    plot_training_loss(cfg.OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to Detectron2 YAML configuration"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Full path to dataset folder"
    )

    parser.add_argument(
        "--output-dir",
        help="Unique output directory for this training run"
    )

    parser.add_argument(
        "--output-root",
        default="outputs/training",
        help="Root directory for training outputs"
    )

    parser.add_argument(
        "--weights",
        choices=["apply", "none"],
        default="apply"
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        required=True
    )

    parser.add_argument(
        "--advanced-config",
        default=""
    )

    args = parser.parse_args()

    main(args)
