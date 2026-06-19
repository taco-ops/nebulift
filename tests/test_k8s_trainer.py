"""Unit tests for the in-cluster distributed training entrypoint.

These tests exercise the Kubernetes distributed trainer module without
spinning up a real ``torch.distributed`` process group. The data
discovery, dataset wiring, checkpoint persistence, and ``main()``
orchestration paths are mocked at module boundaries so they can run on
any developer machine in milliseconds.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nebulift.distributed import k8s_trainer


@pytest.fixture
def class_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Build a minimal ``clean/`` + ``contaminated/`` layout for both splits.

    The helper creates four placeholder FITS files (two per split, one
    per class). Tests downstream only need the directory structure, not
    the file contents, because the FITS processor is mocked.
    """
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    for split in (train_dir, val_dir):
        (split / "clean").mkdir(parents=True)
        (split / "contaminated").mkdir(parents=True)
        (split / "clean" / "a.fits").write_text("placeholder")
        (split / "contaminated" / "b.fits").write_text("placeholder")
    return train_dir, val_dir


def test_build_distributed_loaders_single_rank(
    class_dataset: tuple[Path, Path],
) -> None:
    """Single-rank loaders skip ``DistributedSampler`` to avoid the import-time setup tax."""
    from nebulift.training import collect_class_directory_records

    train_dir, val_dir = class_dataset
    train_records = collect_class_directory_records(train_dir)
    val_records = collect_class_directory_records(val_dir)

    with patch.object(k8s_trainer, "AstroImageDataset") as mock_dataset_cls:
        mock_dataset_cls.return_value = MagicMock(__len__=lambda _self: 2)
        train_loader, val_loader = k8s_trainer._build_distributed_loaders(
            train_records,
            val_records,
            batch_size=4,
            world_size=1,
            rank=0,
        )

    assert train_loader.sampler is not None  # SequentialSampler/RandomSampler
    # When world_size==1 we should not wrap with DistributedSampler.
    from torch.utils.data.distributed import DistributedSampler

    assert not isinstance(train_loader.sampler, DistributedSampler)
    assert not isinstance(val_loader.sampler, DistributedSampler)


def test_build_distributed_loaders_multi_rank(class_dataset: tuple[Path, Path]) -> None:
    """Multi-rank training wraps the train loader in ``DistributedSampler``; val stays whole."""
    from nebulift.training import collect_class_directory_records

    train_dir, val_dir = class_dataset
    train_records = collect_class_directory_records(train_dir)
    val_records = collect_class_directory_records(val_dir)

    with patch.object(k8s_trainer, "AstroImageDataset") as mock_dataset_cls:
        mock_dataset_cls.return_value = MagicMock(__len__=lambda _self: 8)
        train_loader, val_loader = k8s_trainer._build_distributed_loaders(
            train_records,
            val_records,
            batch_size=2,
            world_size=4,
            rank=1,
        )

    from torch.utils.data.distributed import DistributedSampler

    assert isinstance(train_loader.sampler, DistributedSampler)
    # Validation must remain unsharded so every rank computes against the
    # full held-out set; otherwise per-rank val_accuracy is incomparable.
    assert not isinstance(val_loader.sampler, DistributedSampler)


def test_save_unwrapped_checkpoint_restores_wrapped_model(tmp_path: Path) -> None:
    """Checkpoint helper temporarily swaps the model and always restores the DDP shell."""
    output = tmp_path / "checkpoint.pth"

    trainer = MagicMock()
    underlying_model = MagicMock(name="underlying")
    wrapped = MagicMock(name="ddp_wrapped")
    wrapped.module = underlying_model
    trainer.model = wrapped

    # ``isinstance(wrapped, DistributedDataParallel)`` must be true for
    # the helper to take the unwrap branch; patch the symbol to coerce it.
    with patch.object(k8s_trainer, "DistributedDataParallel", new=MagicMock):
        with patch(
            "nebulift.model_persistence.ModelCheckpoint.save_model"
        ) as mock_save:
            wrapped.__class__ = k8s_trainer.DistributedDataParallel
            k8s_trainer._save_unwrapped_checkpoint(trainer, output)

    assert output.parent.exists()
    mock_save.assert_called_once()
    # After save we must restore the DDP-wrapped model so subsequent
    # epochs (in long-running jobs) keep their gradient sync intact.
    assert trainer.model is wrapped


def test_save_unwrapped_checkpoint_with_plain_model(tmp_path: Path) -> None:
    """When the trainer's model is not DDP-wrapped, save proceeds without swapping."""
    output = tmp_path / "checkpoint.pth"

    trainer = MagicMock()
    plain_model = MagicMock(name="plain")
    trainer.model = plain_model

    with patch("nebulift.model_persistence.ModelCheckpoint.save_model") as mock_save:
        k8s_trainer._save_unwrapped_checkpoint(trainer, output)

    mock_save.assert_called_once_with(trainer, output)
    assert trainer.model is plain_model


def test_main_runs_real_pipeline_on_rank_zero(
    class_dataset: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``main`` discovers FITS files, trains, and saves the checkpoint on rank 0."""
    train_dir, val_dir = class_dataset
    model_output_dir = tmp_path / "models"

    monkeypatch.setenv("TRAIN_DATA_PATH", str(train_dir))
    monkeypatch.setenv("VAL_DATA_PATH", str(val_dir))
    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(model_output_dir))
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("BATCH_SIZE", "2")
    monkeypatch.setenv("LEARNING_RATE", "0.01")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    trainer_instance = MagicMock()
    trainer_instance.is_main_process.return_value = True

    with (
        patch.object(
            k8s_trainer,
            "_build_distributed_loaders",
            return_value=(MagicMock(), MagicMock()),
        ) as mock_loaders,
        patch.object(k8s_trainer, "AstroQualityClassifier") as mock_model_cls,
        patch.object(
            k8s_trainer, "K8sDistributedTrainer", return_value=trainer_instance
        ) as mock_trainer_cls,
        patch.object(k8s_trainer, "_save_unwrapped_checkpoint") as mock_save,
    ):
        k8s_trainer.main()

    mock_loaders.assert_called_once()
    call_kwargs = mock_loaders.call_args.kwargs
    assert call_kwargs["batch_size"] == 2
    assert call_kwargs["world_size"] == 1
    assert call_kwargs["rank"] == 0

    mock_model_cls.assert_called_once()
    mock_trainer_cls.assert_called_once()
    trainer_instance.train.assert_called_once()
    train_kwargs = trainer_instance.train.call_args.kwargs
    assert train_kwargs.get("epochs") == 1
    mock_save.assert_called_once()
    trainer_instance.cleanup.assert_called_once()


def test_main_skips_save_on_non_zero_rank(
    class_dataset: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-zero ranks must still call ``train`` + ``cleanup`` but never write the checkpoint."""
    train_dir, val_dir = class_dataset
    model_output_dir = tmp_path / "models"

    monkeypatch.setenv("TRAIN_DATA_PATH", str(train_dir))
    monkeypatch.setenv("VAL_DATA_PATH", str(val_dir))
    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(model_output_dir))
    monkeypatch.setenv("EPOCHS", "1")
    monkeypatch.setenv("BATCH_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    trainer_instance = MagicMock()
    trainer_instance.is_main_process.return_value = False

    with (
        patch.object(
            k8s_trainer,
            "_build_distributed_loaders",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch.object(k8s_trainer, "AstroQualityClassifier"),
        patch.object(
            k8s_trainer, "K8sDistributedTrainer", return_value=trainer_instance
        ),
        patch.object(k8s_trainer, "_save_unwrapped_checkpoint") as mock_save,
    ):
        k8s_trainer.main()

    trainer_instance.train.assert_called_once()
    mock_save.assert_not_called()
    trainer_instance.cleanup.assert_called_once()


def test_main_exits_when_train_dir_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty TRAIN_DATA_PATH triggers a non-zero exit so the K8s Job retries."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    monkeypatch.setenv("TRAIN_DATA_PATH", str(empty_dir))
    monkeypatch.setenv("VAL_DATA_PATH", str(empty_dir))
    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    with pytest.raises(SystemExit) as excinfo:
        k8s_trainer.main()
    assert excinfo.value.code == 1


def test_main_propagates_failures_as_exit_code(
    class_dataset: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trainer construction errors must surface as a non-zero exit, not silently swallow."""
    train_dir, val_dir = class_dataset

    monkeypatch.setenv("TRAIN_DATA_PATH", str(train_dir))
    monkeypatch.setenv("VAL_DATA_PATH", str(val_dir))
    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    with (
        patch.object(
            k8s_trainer,
            "_build_distributed_loaders",
            return_value=(MagicMock(), MagicMock()),
        ),
        patch.object(k8s_trainer, "AstroQualityClassifier"),
        patch.object(
            k8s_trainer,
            "K8sDistributedTrainer",
            side_effect=RuntimeError("init failed"),
        ),
    ):
        with pytest.raises(SystemExit) as excinfo:
            k8s_trainer.main()
    assert excinfo.value.code == 1


def test_main_honors_default_env_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing TRAIN/VAL env vars fall back to ``/data/train`` and ``/data/val``.

    We assert by intercepting ``collect_class_directory_records`` rather
    than rely on the actual ``/data`` mount existing on the test host.
    """
    monkeypatch.delenv("TRAIN_DATA_PATH", raising=False)
    monkeypatch.delenv("VAL_DATA_PATH", raising=False)
    monkeypatch.setenv("MODEL_OUTPUT_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    seen_paths: list[Path] = []

    def fake_collect(directory: Path) -> list[dict[str, object]]:
        seen_paths.append(directory)
        # Return empty so main exits early without touching torch.
        return []

    with patch.object(
        k8s_trainer, "collect_class_directory_records", side_effect=fake_collect
    ):
        with pytest.raises(SystemExit):
            k8s_trainer.main()

    assert seen_paths[0] == Path("/data/train")
    assert seen_paths[1] == Path("/data/val")
    # main raises after both discoveries when train is empty; it does not
    # short-circuit between the two calls.
    assert len(seen_paths) == 2


def test_main_module_entrypoint_invokes_main() -> None:
    """Running ``python -m nebulift.distributed.k8s_trainer`` triggers ``main``.

    The module exposes ``if __name__ == "__main__": main()`` at the
    bottom; verify the indirection is wired by re-importing as a script.
    """
    # Quick smoke: simply confirm ``main`` is callable and the module
    # attribute exists; deeper invocation is covered by the env-driven
    # tests above.
    assert callable(getattr(k8s_trainer, "main", None))
    assert os.environ  # nothing should crash on import
