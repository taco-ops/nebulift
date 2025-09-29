# Nebulift: AI-Powered Astrophotography Quality Control 🚀

[![CI/CD Pipeline](https://github.com/taco-ops/nebulift/actions/workflows/ci.yml/badge.svg)](https://github.com/taco-ops/nebulift/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/taco-ops/nebulift/branch/main/graph/badge.svg)](https://codecov.io/gh/taco-ops/nebulift)
[![Latest Release](https://img.shields.io/github/v/release/taco-ops/nebulift?color=orange&include_prereleases)](https://github.com/taco-ops/nebulift/releases)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://hub.docker.com)
[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io)

*Tired of manually sorting through thousands of astrophotos to find the keepers?* **Nebulift** is here to help! 🌟

This **beta** system automatically identifies and filters out contaminated astronomical images using a hybrid approach: traditional computer vision for artifact detection + ResNet18 deep learning for quality assessment. Built specifically for telescope data in FITS format, it can run anywhere from your laptop to a Raspberry Pi 5 cluster.

**🧪 Beta Status**: Core functionality is complete with enterprise-grade infrastructure, but we're actively seeking real-world testing and feedback from the astrophotography community before our 1.0 release.

## 🎯 Why Nebulift?

**The Problem**: After a night of imaging, you're left with hundreds or thousands of photos. Some are crystal clear, others have satellite streaks, airplane trails, clouds, or other artifacts that would ruin your final stack. Manually sorting through them all? Ain't nobody got time for that! 😅

**The Solution**: Nebulift automatically analyzes your images and sorts them into three categories:
- ✨ **Clean**: Ready for stacking 
- 🗑️ **Contaminated**: Save yourself the headache, skip these
- 🤔 **Review**: Borderline cases that might need a human eye

## 🚀 What Makes It Special?

- **🔭 FITS-Native**: Built specifically for astronomical data (though it handles regular images too)
- **💻 CPU-Only**: No expensive GPU needed - runs great on Raspberry Pi 5!
- **🤖 Hybrid Intelligence**: CV algorithms + ResNet18 = better accuracy than either alone
- **🏗️ Enterprise Infrastructure**: Comprehensive CI/CD, testing, and container deployment
- **📈 Scalable**: From single images to distributed training on Kubernetes clusters
- **🎛️ Semi-Automated**: Generate training labels from CV analysis, then train custom models

## �️ Quick Start (Get Running in 2 Minutes!)

**Prerequisites**: Python 3.9+ and either `uv` (recommended) or `pip`

### Installation
```bash
# Clone and enter the project
git clone https://github.com/taco-ops/nebulift.git
cd nebulift

# Option 1: Using uv (faster, recommended)
pip install uv  # If you don't have uv yet
uv sync         # Install everything

# Option 2: Traditional pip approach
pip install -e .
```

### 🎯 Try It Out!

**Single Image Analysis** (great for testing):
```bash
# Analyze a single FITS file
uv run nebulift analyze my_awesome_nebula.fits

# Or with a regular image
uv run nebulift analyze moon_shot.jpg
```

**Batch Processing** (the real power):
```bash
# Sort an entire night's worth of images
uv run nebulift batch /path/to/raw/images /path/to/sorted/output

# This creates three folders:
# - clean/        <- Your best images, ready for stacking!
# - contaminated/ <- Skip these ones
# - review/       <- Borderline cases to check manually
```

**Want to get your hands dirty with the code?** Here's how:

```python
from nebulift.fits_processor import FITSProcessor
from nebulift.cv_prefilter import ArtifactDetector

# Initialize the components
processor = FITSProcessor()
detector = ArtifactDetector()

# Load and process a FITS file
fits_data = processor.load_fits_file("your_image.fits")
normalized = processor.normalize_image(fits_data['image_data'])

# Run the analysis
analysis = detector.comprehensive_analysis(normalized)

print(f"Quality score: {analysis['overall_quality_score']:.3f}")
print(f"Has streaks: {analysis['streaks']['has_streaks']}")
print(f"Recommended action: {'Keep' if analysis['overall_quality_score'] > 0.7 else 'Review' if analysis['overall_quality_score'] > 0.3 else 'Discard'}")
```

## 🧠 Train Your Own Models (Advanced Users)

Got specific needs? Train a custom model on your own data! The system makes this surprisingly straightforward:

**One-Command Training Pipeline**:
```bash
# Train a model from your FITS files (uses CV analysis to generate labels)
uv run nebulift train-from-fits /path/to/telescope/session/ \
    --model_output models/my_custom_classifier.pth \
    --dataset_dir datasets/organized \
    --epochs 50 \
    --clean_threshold 0.8  # How picky should we be?
```

This will:
1. Analyze all your images with computer vision algorithms
2. Generate training labels based on quality scores
3. Organize files into training/validation sets
4. Train a ResNet18 model
5. Save everything for future use

**Scale Up with Kubernetes** (for the truly ambitious):
```bash
# Deploy to your Raspberry Pi 5 cluster
docker build -t nebulift:latest .
kubectl apply -f k8s/
kubectl logs -f job/nebulift-training

# Need more power? Scale it up!
kubectl patch job nebulift-training -p '{"spec":{"parallelism":8}}'
```

## 🏗️ Architecture (For the Curious)

**Four-Component Pipeline**:
1. **FITS Processor**: Loads astronomical files, handles metadata, normalizes data
2. **CV Pre-filter**: Traditional computer vision (Hough transforms, texture analysis)
3. **ML Model**: ResNet18 binary classifier for final quality assessment
4. **Distributed Training**: Kubernetes-native scaling for big datasets

**The Magic Workflow**:
```
Raw FITS Files → Normalization → CV Analysis → Quality Scoring → ML Classification
                                      ↓
                              Auto-Generated Labels → Model Training → Deployment
```

## 🚦 Project Status

**Current State**: Beta - Ready for Real-World Testing! 🧪
- ✅ **67 passing tests** with comprehensive coverage
- ✅ **Enterprise CI/CD pipeline** with multi-platform testing  
- ✅ **Container images** ready for deployment
- ✅ **Complete documentation** and examples
- ✅ **Model persistence** with versioning and metadata
- ✅ **Distributed training** infrastructure

**🔬 Beta Testing Phase**:
We're actively seeking feedback from the astrophotography community! The core system is feature-complete with robust infrastructure, but we want to validate performance across different:
- 🔭 Telescope setups and imaging conditions
- 📊 Various FITS file formats and metadata structures  
- 🌍 Real observatory workflows and data volumes
- ⚙️ Hardware configurations (from laptops to RPi5 clusters)

**What's Next** (Post-Beta):
- 📊 Real-world performance benchmarks on various telescopes
- 🎛️ GUI interface for non-command-line users
- 📱 Integration with popular astrophotography software
- 🌟 More sophisticated artifact detection algorithms

## 🧪 Validation & Testing

**Want to make sure everything's working?** Run the validation:
```bash
uv run python validate_system.py  # Full end-to-end test
uv run pytest -v                  # Run the test suite (67 tests!)
uv run python test_model_persistence.py  # Test model save/load
```

**Test Coverage**:
- **54 core tests + 13 model persistence tests** = 67 total
- **89% code coverage** across the main components
- **Property-based testing** with Hypothesis for edge cases
- **Multi-platform CI** (Ubuntu, macOS, Python 3.9-3.12)

### Hardware Requirements
- **Minimum**: 4GB RAM, modern CPU
- **Recommended**: 8GB RAM for large batches
- **Raspberry Pi 5**: Fully supported (CPU-only mode)
- **GPU**: Optional (CPU mode is default and recommended)
- **Distributed**: Kubernetes cluster with NFS storage for multi-node training

### Distributed Training Architecture

The system supports distributed training across Kubernetes clusters using PyTorch's DistributedDataParallel:

#### Components
- **K8sDistributedTrainer**: Extends base ModelTrainer with distributed capabilities
- **Data Sharding**: Automatic dataset distribution across training nodes
- **Model Aggregation**: Synchronous gradient aggregation using all-reduce operations
- **Kubernetes Integration**: Native K8s job orchestration with persistent storage

## 🔬 Technical Highlights (For the Tech-Savvy)

**System Architecture**:
- **Modular Design**: Four independent components that work together seamlessly
- **CPU-Optimized**: PyTorch quantization + efficient algorithms = no GPU needed
- **Container-Native**: Multi-arch Docker images (AMD64 + ARM64) ready for deployment
- **K8s Integration**: Full Kubernetes support with persistent storage and resource management

**Advanced Features**:
- **Model Persistence**: Comprehensive checkpoint system with versioning and metadata
- **Distributed Training**: PyTorch DistributedDataParallel with 'gloo' backend for CPU clusters
- **Smart Caching**: Image statistics caching for improved performance
- **Robust Error Handling**: Graceful degradation when files are corrupted
- **Property-Based Testing**: Hypothesis framework catches edge cases you'd never think of

**Code Quality** (because we care about maintainable software):
- **Type Safety**: Full mypy type checking with strict enforcement
- **Linting**: Black + Ruff + pre-commit hooks = consistent, clean code
- **Testing**: 67 tests with 89% coverage + comprehensive CI/CD pipeline
- **Documentation**: Every function documented, examples everywhere

## � Beta Testing - We Need Your Help!

**Are you an astrophotographer?** We'd love your feedback! Here's how you can help us make Nebulift better:

### 🔬 What We're Looking For
- **Real telescope data**: Test with your actual FITS files from different telescopes/cameras
- **Edge cases**: Unusual lighting conditions, rare artifacts, specific telescope configurations
- **Performance feedback**: How does it run on your hardware? (Especially Raspberry Pi setups!)
- **Workflow integration**: Does it fit into your existing image processing pipeline?

### 📊 Easy Ways to Contribute
1. **Try it out**: Download and test on a small batch of your images
2. **Report results**: Open an issue with your experience (good or bad!)
3. **Share data**: If willing, share problematic images that don't classify correctly
4. **Suggest features**: What would make this more useful for your workflow?

### 🎯 Beta Testing Goals
- Validate accuracy across different telescope/camera combinations
- Optimize performance for various hardware configurations
- Identify edge cases and improve artifact detection
- Gather user experience feedback for UI/workflow improvements

**Contact**: Open a GitHub issue or discussion - we're actively monitoring and will respond quickly!

## �🤝 Contributing & Community

**Found a bug?** Open an issue! **Have an idea?** We'd love to hear it! **Want to contribute?** PRs are welcome!

This project is designed to grow with the astrophotography community. Whether you're:
- 🔭 An observatory operator with specific needs
- 🧑‍💻 A developer who wants to improve the algorithms  
- 📊 A data scientist interested in astronomical ML applications
- 🏭 A systems admin deploying on Kubernetes clusters

There's a place for you here!

## 🙏 Acknowledgments

Built with love for the astrophotography community. Special thanks to:
- The **PyTorch** team for making distributed training accessible
- **Astropy** developers for excellent FITS file handling
- The **Kubernetes** ecosystem for making container orchestration smooth
- Everyone who's ever spent a cold night under the stars capturing photons ✨

## � License

MIT License - feel free to use this however you'd like! If it helps you capture better images of the cosmos, we've done our job. 🌌

---

*Happy imaging, and may your nights be clear and your satellites be few!* 🚀🌟  
3. **Performance Validation**: Benchmark on Raspberry Pi 5 clusters and resource-constrained hardware
4. **CLI Interface**: Command-line tool for batch processing and distributed training workflows
5. **Production Deployment**: Helm charts and production-ready K8s deployments
6. **Distributed Inference**: Extend distributed capabilities to inference workloads

Once validated through real-world use and cluster deployments, the system will be promoted to production status.

## 📄 License & Credits

This project demonstrates professional software development practices for scientific applications with distributed computing capabilities. The implementation follows modern Python standards and is designed for real-world astronomical image processing workflows across single nodes and Kubernetes clusters.

**Note**: This is a beta implementation with core functionality and distributed training complete with comprehensive testing. Real-world validation and performance optimization on actual telescope data and RPi5 clusters are needed before production deployment. All core requirements have been met and the system has been thoroughly unit tested.
