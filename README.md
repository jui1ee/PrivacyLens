# PrivacyLens: Task-Aware Visual Privacy Protection with Human-in-the-Loop Redaction

## Setup & Running Guide

### 1. Environment Requirements
- Python 3.10+ (tested on Python 3.12.5 with CUDA 12.4 on local NVIDIA GTX 1650 4GB).
- Compatible with local Windows / Linux and Google Colab (T4 GPU).

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
VizWiz-Priv annotations are automatically parsed from `data/raw/Annotations/Annotations/`.
Images are streamed selectively on demand via HTTP Range requests from the official repository or read from local cache (`data/cache/`).

### 3. Executing the Training Notebook Headlessly
To run the research notebook end-to-end and preserve real outputs in the notebook file:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/privacylens_training.ipynb
```
Outputs and figures are saved to `results/` and `results/figures/`.
