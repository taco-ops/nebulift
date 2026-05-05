# Nebulift: Astrophotography Quality Control

[![CI/CD Pipeline](https://github.com/taco-ops/nebulift/actions/workflows/ci.yml/badge.svg)](https://github.com/taco-ops/nebulift/actions/workflows/ci.yml)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/taco-ops/nebulift/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/taco-ops/nebulift/tree/main)
[![codecov](https://codecov.io/gh/taco-ops/nebulift/branch/main/graph/badge.svg)](https://codecov.io/gh/taco-ops/nebulift)
[![Latest Release](https://img.shields.io/github/v/release/taco-ops/nebulift?color=orange&include_prereleases)](https://github.com/taco-ops/nebulift/releases)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://hub.docker.com)
[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io)

Nebulift is a beta Python tool for quality assessment of astronomical FITS images. It uses traditional computer vision to detect common acquisition artifacts and can train a ResNet18 classifier from CV-generated labels.

The current focus is a reliable local CLI workflow for FITS files. Kubernetes, Argo CD, and distributed training resources are present for future deployment work, but they should be treated as scaffolding until validated with real datasets and cluster storage.

## Use Case

Astrophotography sessions often produce many FITS frames that need to be reviewed before stacking. Nebulift helps identify likely clean, contaminated, and borderline frames so users can reduce manual inspection time.

Current categories:

- `clean`: likely suitable for stacking
- `contaminated`: likely affected by streaks, cloud cover, saturation, hot pixels, or other artifacts
- `review`: borderline cases that should be inspected manually

## Current Status

Implemented:

- FITS loading, metadata extraction, normalization, and ML-ready resizing
- Computer vision artifact detection for streaks, clouds, saturation, hot pixels, and quality scoring
- Batch analysis of FITS files with JSON manifest output
- Three-class CV-generated labels: `contaminated`, `clean`, and `review`
- Optional ML-assisted classification for `analyze` and `batch` when a model checkpoint is provided
- Optional batch moving into class buckets with `--action move`
- Interactive terminal review of batch manifests
- Local FITS-to-model training pipeline through `nebulift train-from-fits`
- Model checkpointing, metadata, and versioning utilities
- Docker, Kubernetes, Kustomize, Argo CD, GitHub Actions, and CircleCI configuration

Not yet complete:

- Batch review currently updates JSON manifests only; it does not display images or write curated datasets
- Regular image analysis is supported by lower-level CV utilities, but the primary CLI workflow is FITS-oriented
- No pretrained Nebulift model is currently shipped with the repository
- Distributed Kubernetes training still needs real dataset mounting, model persistence, and end-to-end validation

## Installation

Prerequisites:

- Python 3.9 or newer
- `uv` is recommended for local development

```bash
git clone https://github.com/taco-ops/nebulift.git
cd nebulift

pip install uv
uv sync --all-extras --dev
```

For a minimal editable install without development tooling:

```bash
pip install -e .
```

## CLI Usage

### Analyze One FITS File

```bash
uv run nebulift analyze /path/to/image.fits
```

This runs CV-based analysis and reports a quality score, detected artifacts, review status, CV label, and final decision label.

Use a trained model checkpoint for ML-assisted classification:

```bash
uv run nebulift analyze /path/to/image.fits \
    --model models/custom_classifier.pth
```

### Batch Sort FITS Files

```bash
uv run nebulift batch /path/to/raw/fits /path/to/sorted/output
```

By default, `batch` is report-only. It writes `/path/to/sorted/output/batch_manifest.json` and does not copy or move source files.

Use `--action move` to move files into class buckets:

```bash
uv run nebulift batch /path/to/raw/fits /path/to/sorted/output \
    --action move
```

Move mode creates:

- `clean/`
- `contaminated/`
- `review/`

Use a trained model checkpoint during batch classification:

```bash
uv run nebulift batch /path/to/raw/fits /path/to/sorted/output \
    --model models/custom_classifier.pth
```

When `--model` is provided and inference succeeds, the model label becomes the decision label. The manifest still records the CV label, quality score, decision source, artifact flags, and model prediction.

### Review a Batch Manifest

```bash
uv run nebulift review /path/to/sorted/output/batch_manifest.json
```

The review command prompts for corrected labels in the terminal and updates the manifest in place. It currently supports label correction only; it does not display image previews.

### Train From FITS Files

```bash
uv run nebulift train-from-fits /path/to/fits/session \
    --model_output models/custom_classifier.pth \
    --dataset_dir datasets/session_training \
    --epochs 50 \
    --batch_size 32 \
    --clean_threshold 0.7 \
    --contaminated_threshold 0.3
```

This workflow:

1. Finds FITS files in the input directory.
2. Processes each file with `FITSProcessor`.
3. Runs CV artifact detection with `ArtifactDetector`.
4. Assigns one of three labels based on quality thresholds.
5. Writes train and validation manifests into `--dataset_dir`.
6. Trains a three-class ResNet18 model.
7. Saves the model checkpoint and metadata to `--model_output`.

Threshold behavior:

- `score >= clean_threshold`: `clean`
- `score <= contaminated_threshold`: `contaminated`
- scores between thresholds: `review`

## Python API Example

```python
from pathlib import Path

from nebulift.ml_model import complete_training_pipeline

results = complete_training_pipeline(
    fits_directory=Path("/data/fits/session"),
    model_output_path=Path("models/session_classifier.pth"),
    dataset_output_dir=Path("datasets/session_training"),
    epochs=20,
    batch_size=16,
    clean_threshold=0.7,
    contaminated_threshold=0.3,
)

print(results["final_metrics"])
print(results["dataset_stats"])
```

## Validation

```bash
uv run pytest tests/ -v
uv run python test_training_pipeline.py
uv run python test_model_persistence.py
uv run black --check --diff .
uv run flake8 nebulift/ tests/
uv run mypy nebulift/ --ignore-missing-imports
```

## CircleCI

Nebulift keeps source code in GitHub and can run validation from CircleCI using `.circleci/config.yml`.

Regular CircleCI pipelines run formatting, linting, type checks, unit tests, model persistence checks, and CLI training command checks.

The full FITS training pipeline smoke test is available through the manual `run-training-pipeline=true` pipeline parameter. Use it when validating training changes because it is more expensive than the baseline checks.

## Deployment Resources

The repository includes Kubernetes and Argo CD resources for future distributed training workflows:

- `k8s/`: base manifests and Kustomize overlays
- `argocd/`: Argo CD Application, ApplicationSet, and AppProject resources
- `nebulift/distributed/`: PyTorch distributed training utilities

These resources are useful for infrastructure iteration, but the local training pipeline is the primary supported path today.

## Hardware Notes

- Minimum local use: modern CPU and 4 GB RAM
- Recommended for training: 8 GB RAM or more
- GPU is optional; the project defaults to CPU-compatible behavior
- Raspberry Pi 5 support is a design target, but real-world performance validation is still needed

## Contributing

Useful areas for contribution:

- Real FITS dataset validation
- Curated-label import from reviewed JSON manifests
- Pretrained model packaging or download workflow
- Distributed training integration with real storage
- CLI integration tests
- Documentation corrections based on real deployments

Open an issue or pull request with the dataset, environment, command, and observed behavior whenever possible.

## License

MIT License. See the repository license for details.
