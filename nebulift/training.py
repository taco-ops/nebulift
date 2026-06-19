"""Training workflow helpers for Nebulift CLI commands."""

import random
from pathlib import Path
from typing import Any, Callable, Optional

from torch.utils.data import DataLoader

from .fits_processor import FITSProcessor
from .ml_model import (
    LABEL_IDS,
    AstroImageDataset,
    AstroQualityClassifier,
    ModelTrainer,
    complete_training_pipeline,
    complete_training_pipeline_from_manifest,
    create_data_transforms,
)

ProgressCallback = Callable[[int, int, str], None]


def collect_class_directory_records(data_dir: Path) -> list[dict[str, Any]]:
    """Walk a class-folder dataset and return ``{path, label, label_name}`` records.

    The dataset is expected to contain one subdirectory per class in
    :data:`nebulift.ml_model.LABEL_IDS` (``clean``, ``contaminated``,
    ``review``). FITS files are discovered recursively within each class
    subdirectory. Missing class directories are silently skipped, which
    allows callers to operate on partial datasets.

    Args:
        data_dir: Root directory containing per-class subdirectories.

    Returns:
        Sorted list of record dictionaries, deterministic for a given
        directory layout.
    """
    records = []
    patterns = ["*.fits", "*.fit", "*.fts"]
    for label_name, label_id in LABEL_IDS.items():
        label_dir = data_dir / label_name
        if not label_dir.exists():
            continue
        for pattern in patterns:
            for file_path in label_dir.rglob(pattern):
                records.append(
                    {
                        "path": file_path,
                        "label": label_id,
                        "label_name": label_name,
                    },
                )
    return sorted(records, key=lambda record: str(record["path"]))


def _collect_class_directory_records(data_dir: Path) -> list[dict[str, Any]]:
    # Deprecated private alias; kept for backwards compatibility within
    # this module. New callers should use ``collect_class_directory_records``.
    return collect_class_directory_records(data_dir)


def train_model(
    data_dir: Path,
    model_output: Path,
    epochs: int = 50,
    batch_size: int = 32,
    train_split: float = 0.8,
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    """Train the ML model locally from curated class folders."""
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0.0 and 1.0")

    records = _collect_class_directory_records(data_dir)
    if len(records) < 2:
        raise ValueError(
            "At least two FITS files are required under clean/, contaminated/, or review/",
        )

    random.Random(42).shuffle(records)  # nosec B311 - deterministic split only
    split_index = int(len(records) * train_split)
    split_index = min(max(split_index, 1), len(records) - 1)
    train_records = records[:split_index]
    val_records = records[split_index:]

    processor = FITSProcessor()
    train_dataset = AstroImageDataset(
        [record["path"] for record in train_records],
        [record["label"] for record in train_records],
        transform=create_data_transforms(train=True),
        fits_processor=processor,
    )
    val_dataset = AstroImageDataset(
        [record["path"] for record in val_records],
        [record["label"] for record in val_records],
        transform=create_data_transforms(train=False),
        fits_processor=processor,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = AstroQualityClassifier(num_classes=3, pretrained=False)
    trainer = ModelTrainer(model)
    trainer.train(
        train_loader,
        val_loader,
        epochs=epochs,
        progress_callback=progress_callback,
    )
    trainer.save_model(model_output)

    print("Training complete!")
    print(f"Model saved to: {model_output}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")


def train_from_fits(
    fits_dir: Path,
    model_output: Path,
    dataset_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    clean_threshold: float = 0.7,
    contaminated_threshold: float = 0.3,
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    """Train model using CV-generated labels from FITS files."""
    print(f"Starting complete training pipeline from {fits_dir}")
    print(
        f"Clean threshold: {clean_threshold}, Contaminated threshold: {contaminated_threshold}",
    )
    results = complete_training_pipeline(
        fits_directory=fits_dir,
        model_output_path=model_output,
        dataset_output_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        clean_threshold=clean_threshold,
        contaminated_threshold=contaminated_threshold,
        progress_callback=progress_callback,
    )

    stats = results["dataset_stats"]
    metrics = results["final_metrics"]
    print("\nTraining complete!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Dataset manifests saved to: {results['dataset_dir']}")
    print(f"Training samples: {stats['training_samples']}")
    print(f"Validation samples: {stats['validation_samples']}")
    print(f"Review samples: {stats['review_samples']}")
    print(f"Best validation accuracy: {metrics['best_val_accuracy']:.2f}%")


def train_from_manifest(
    manifest_path: Path,
    model_output: Path,
    dataset_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    reviewed_only: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> None:
    """Train model using labels from a reviewed JSON batch manifest."""
    print(f"Starting training pipeline from manifest {manifest_path}")
    results = complete_training_pipeline_from_manifest(
        manifest_path=manifest_path,
        model_output_path=model_output,
        dataset_output_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        reviewed_only=reviewed_only,
        progress_callback=progress_callback,
    )

    stats = results["dataset_stats"]
    metrics = results["final_metrics"]
    print("\nTraining complete!")
    print(f"Model saved to: {results['model_path']}")
    print(f"Dataset manifests saved to: {results['dataset_dir']}")
    print(f"Training samples: {stats['training_samples']}")
    print(f"Validation samples: {stats['validation_samples']}")
    print(f"Reviewed samples: {stats['reviewed_samples']}")
    print(f"Best validation accuracy: {metrics['best_val_accuracy']:.2f}%")
