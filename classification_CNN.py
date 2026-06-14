import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet


DATASETS = {"KIRC": "TCGA-KIRC", "LUAD": "TCGA-LUAD", "PRAD": "TCGA-PRAD"}
DEFAULT_CLASS_NAMES = {
    0: "Upregulated",
    1: "Downregulated",
    2: "NoDifference",
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    sample_idx: int
    fold: int


class CsvImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(record.path).convert("L")
        return self.transform(image), record.label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 classifiers on {Converter}_{dataset}_IMGS folders."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Workspace root containing *_IMGS folders.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        action="append",
        help="Specific image folder to train. Can be repeated. Defaults to all *_IMGS folders.",
    )
    parser.add_argument("--fold", type=int, default=None, help="Train only this fold.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cnn_results"),
        help="Directory for checkpoints and metrics.",
    )
    parser.add_argument(
        "--hacnet-weights",
        type=Path,
        default=None,
        help="Optional HACNet critic/model/checkpoint weights. Used only for HACNet folders.",
    )
    parser.add_argument(
        "--no-auto-hacnet-weights",
        action="store_true",
        help="Do not auto-load latest HACNet/mlruns critic.pth for HACNet folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests, folds, split counts, and HACNet weight discovery without training.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def discover_image_roots(root) -> Path:
    return sorted(path for path in root.glob("*_IMGS") if path.is_dir())


def split_image_root_name(image_root):
    name = image_root.name
    if not name.endswith("_IMGS") or "_" not in name[:-5]:
        raise ValueError(f"Invalid image folder name: {image_root}")
    converter, dataset = name[:-5].split("_", 1)
    return converter, dataset


def read_manifest(image_root):
    metadata_path = image_root / "metadata.csv"
    labels_path = image_root / "labels.csv"

    # if metadata_path.exists():
    #     records = read_hacnet_metadata(image_root, metadata_path)
    if labels_path.exists():
        records = read_labels_csv(image_root, labels_path)
    else:
        fold_labels = sorted(image_root.glob("fold_*/labels.csv"))
        if not fold_labels:
            raise FileNotFoundError(
                f"{image_root} must contain metadata.csv, labels.csv, or fold_*/labels.csv"
            )
        records = []
        for path in fold_labels:
            records.extend(read_labels_csv(image_root, path))

    missing = [record.path for record in records if not record.path.exists()]
    if missing:
        examples = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"{image_root} has missing image paths, examples: {examples}")

    if not records:
        raise ValueError(f"{image_root} has no image records.")

    return records


def read_hacnet_metadata(image_root, metadata_path):
    records = []
    with metadata_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"sample_idx", "fold", "label", "image_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{metadata_path} must contain {sorted(required)}")
        for row in reader:
            relative_path = Path(row["image_path"])
            if len(relative_path.parts) >= 2 and relative_path.parts[0] == "datasets":
                relative_path = Path(*relative_path.parts[2:])
            path = image_root / relative_path.relative_to(relative_path.parts[0])
            records.append(
                ImageRecord(
                    path=path,
                    label=int(row["label"]),
                    sample_idx=int(row["sample_idx"]),
                    fold=int(row["fold"]),
                )
            )
    return records


def read_labels_csv(image_root, labels_path):
    records = []
    with labels_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"sample_idx", "fold", "label", "path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{labels_path} must contain {sorted(required)}")
        for row in reader:
            path = Path(row["path"])
            if not path.is_absolute():
                path = image_root.parent / path
            records.append(
                ImageRecord(
                    path=path,
                    label=int(row["label"]),
                    sample_idx=int(row["sample_idx"]),
                    fold=int(row["fold"]),
                )
            )
    return records


def get_folds(records):
    return sorted({record.fold for record in records})


def make_fold_split(records, fold):
    train_records = [record for record in records if record.fold == fold]
    train_sample_idxs = {record.sample_idx for record in train_records}

    candidate_test_records = [
        record for record in records if record.sample_idx not in train_sample_idxs
    ]
    by_sample = {}
    for record in sorted(candidate_test_records, key=lambda item: (item.sample_idx, item.fold)):
        by_sample.setdefault(record.sample_idx, record)

    test_records = list(by_sample.values())
    if not train_records:
        raise ValueError(f"Fold {fold} has no train records.")
    if not test_records:
        raise ValueError(
            f"Fold {fold} has no complement test records. Check fold assignment semantics."
        )

    return train_records, test_records


def stratified_train_val_split(records, val_size, seed):
    if not 0 < val_size < 1:
        return list(records), []

    rng = random.Random(seed)
    by_label = defaultdict(list)
    for record in records:
        by_label[record.label].append(record)

    train_records = []
    val_records = []
    for label_records in by_label.values():
        rng.shuffle(label_records)
        val_count = max(1, int(round(len(label_records) * val_size)))
        if val_count >= len(label_records):
            val_count = max(0, len(label_records) - 1)
        val_records.extend(label_records[:val_count])
        train_records.extend(label_records[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def make_transforms(image_size):
    train_transform = transforms.Compose(
        [
           # transforms.Resize((image_size, image_size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    eval_transform = transforms.Compose(
        [
    #       transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return train_transform, eval_transform


def make_loader(records, transform, batch_size, shuffle, num_workers):
    dataset = CsvImageDataset(records, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_resnet18(num_classes) -> ResNet:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def latest_hacnet_weights(root, dataset):
    tcga_dataset = DATASETS.get(dataset, f"TCGA-{dataset}")
    candidates = []
    for path in (root / "HACNet" / "mlruns").glob(f"*-{tcga_dataset}/*/artifacts/critic.pth"):
        candidates.append(path)
    for path in (root / "HACNet" / "mlruns").glob("*/" "*/artifacts/critic.pth"):
        params_path = path.parents[1] / "params" / "dataset"
        if params_path.exists() and params_path.read_text().strip() == tcga_dataset:
            candidates.append(path)
    if not candidates:
        return None
    return max(set(candidates), key=lambda path: path.stat().st_mtime)


def load_hacnet_weights(model, weights_path, device):
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"{weights_path} does not contain a PyTorch state dict.")

    model_state = model.state_dict()
    extracted = {}
    for key, value in state_dict.items():
        clean_key = key.removeprefix("module.")
        for prefix in ("critic.", "model.critic."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            extracted[clean_key] = value

    if not extracted:
        raise ValueError(f"No matching ResNet18 critic weights found in {weights_path}.")

    model.load_state_dict(extracted, strict=False)
    return sorted(extracted)


def classification_metrics(labels, predictions, probabilities, num_classes, loss):
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "roc_auc": None,
    }

    try:
        if num_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(labels, probabilities[:, 1]))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    labels,
                    probabilities,
                    labels=list(range(num_classes)),
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError:
        metrics["roc_auc"] = None

    return metrics


def train_one_epoch(
    model: ResNet,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.AdamW,
    device,
    num_classes,
):
    model.train()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        probabilities = torch.softmax(logits.detach(), dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    return classification_metrics(
        all_labels,
        all_predictions,
        torch.tensor(all_probabilities).numpy(),
        num_classes,
        total_loss / total,
    )


@torch.inference_mode()
def evaluate(model: ResNet, loader: DataLoader, criterion: nn.CrossEntropyLoss, device, num_classes):
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        probabilities = torch.softmax(logits, dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    return classification_metrics(
        all_labels,
        all_predictions,
        torch.tensor(all_probabilities).numpy(),
        num_classes,
        total_loss / total,
    )


def label_summary(records):
    counts = defaultdict(int)
    for record in records:
        counts[record.label] += 1
    return {
        str(label): {
            "name": DEFAULT_CLASS_NAMES.get(label, f"class_{label}"),
            "count": count,
        }
        for label, count in sorted(counts.items())
    }


def run_fold(args, image_root, records, fold):
    converter, dataset = split_image_root_name(image_root)
    train_records, test_records = make_fold_split(records, fold)
    train_records, val_records = stratified_train_val_split(
        train_records, args.val_size, args.seed
    )
    labels = sorted({record.label for record in records})
    num_classes = max(labels) + 1

    train_transform, eval_transform = make_transforms(args.image_size)
    train_loader = make_loader(
        train_records, train_transform, args.batch_size, True, args.num_workers
    )
    val_loader = (
        make_loader(val_records, eval_transform, args.batch_size, False, args.num_workers)
        if val_records
        else None
    )
    test_loader = make_loader(
        test_records, eval_transform, args.batch_size, False, args.num_workers
    )

    device = torch.device(args.device)
    model = build_resnet18(num_classes).to(device)
    loaded_weights = None
    if converter == "HACNet":
        weights_path = args.hacnet_weights
        if weights_path is None and not args.no_auto_hacnet_weights:
            weights_path = latest_hacnet_weights(args.root, dataset)
        if weights_path is not None:
            loaded_keys = load_hacnet_weights(model, weights_path, device)
            loaded_weights = {
                "path": str(weights_path),
                "matched_keys": len(loaded_keys),
            }

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    run_name = f"{converter}_{dataset}_fold_{fold}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_resnet18.pth"
    last_path = output_dir / "last_resnet18.pth"

    start = time()
    best_metric = -1.0
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, num_classes
        )
        val_metrics = (
            evaluate(model, val_loader, criterion, device, num_classes)
            if val_loader
            else train_metrics
        )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
        )
        if val_metrics["accuracy"] >= best_metric:
            best_metric = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "converter": converter,
                    "dataset": dataset,
                    "fold": fold,
                    "num_classes": num_classes,
                    "epoch": epoch,
                    "validation": val_metrics,
                    "loaded_hacnet_weights": loaded_weights,
                },
                best_path,
            )
        print(
            f"{run_name} epoch {epoch:03d}/{args.epochs} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "converter": converter,
            "dataset": dataset,
            "fold": fold,
            "num_classes": num_classes,
            "epoch": args.epochs,
            "test": test_metrics,
            "loaded_hacnet_weights": loaded_weights,
        },
        last_path,
    )

    result = {
        "run_name": run_name,
        "converter": converter,
        "dataset": dataset,
        "fold": fold,
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "test_count": len(test_records),
        "labels": label_summary(records),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_metric,
        "test": test_metrics,
        "loaded_hacnet_weights": loaded_weights,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "elapsed_seconds": time() - start,
        "history": history,
    }
    with (output_dir / "metrics.json").open("w") as file:
        json.dump(result, file, indent=2)
    print(
        f"{run_name} done test_acc={test_metrics['accuracy']:.4f} "
        f"test_f1={test_metrics['f1_score']:.4f} test_loss={test_metrics['loss']:.4f}"
    )
    return result


def dry_run_image_root(args, image_root, records):
    converter, dataset = split_image_root_name(image_root)
    weights_path = None
    if converter == "HACNet" and not args.no_auto_hacnet_weights:
        weights_path = args.hacnet_weights or latest_hacnet_weights(args.root, dataset)

    print(f"{image_root.name}: records={len(records)} folds={get_folds(records)}")
    if converter == "HACNet":
        print(f"{image_root.name}: hacnet_weights={weights_path}")

    folds = [args.fold] if args.fold is not None else get_folds(records)
    for fold in folds:
        train_records, test_records = make_fold_split(records, fold)
        train_records, val_records = stratified_train_val_split(
            train_records, args.val_size, args.seed
        )
        print(
            f"{image_root.name} fold_{fold}: "
            f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
        )


def main():
    args = parse_args()
    args.root = args.root.resolve()
    args.output_dir = args.output_dir.resolve()
    set_seed(args.seed)

    image_roots = args.image_root or discover_image_roots(args.root)
    if not image_roots:
        raise FileNotFoundError(f"No *_IMGS folders found under {args.root}")

    for image_root in image_roots:
        image_root = image_root.resolve()
        records = read_manifest(image_root)
        if args.dry_run:
            dry_run_image_root(args, image_root, records)
            continue
        converter, dataset = split_image_root_name(image_root)
        run_results = []
        folds = [args.fold] if args.fold is not None else get_folds(records)
        for fold in folds:
            run_results.append(run_fold(args, image_root, records, fold))
        summary_path = args.output_dir / f"{converter}_{dataset}_summary.json"
        with summary_path.open("w") as file:
            json.dump(run_results, file, indent=2)

    if args.dry_run:
        return


if __name__ == "__main__":
    main()
