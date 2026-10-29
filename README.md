# PrivacyLens: Task-Aware Visual Privacy Protection with Human-in-the-Loop Redaction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

**PrivacyLens** is an intelligent, task-aware visual privacy redaction pipeline designed to protect user privacy in visual question answering (VQA) and assistive vision systems (such as assisting people who are blind or low vision).

Rather than blindly blurring entire images or discarding sensitive content essential to answering user inquiries, PrivacyLens combines **vision-language open-vocabulary grounding (Microsoft Florence-2)**, **custom fine-tuned object detection (Faster R-CNN)**, and **OCR extraction** with **task-aware query analysis**. It dynamically preserves information needed to answer the user's question while blacking out unneeded private, personal, or financial information (credit cards, cash, identity cards, sensitive documents, and personal details).

---

### Visual Comparison & Output Demo

Below is an interactive demonstration from the **PrivacyLens Streamlit Dashboard** showing the question-aware privacy redaction in action:

<div align=center>
  <img src=docs/images/demo_redaction_comparison.png alt=PrivacyLens Visual Redaction Demo width=950 />
</div>

<br/>

- **Left (Grounding Overlay)**:
  - 🟩 **Green Box**: Content preserved because it is relevant to the user query (e.g., *What is in the cup?* identifies the coffee cup).
  - 🟥 **Red Boxes**: Unneeded sensitive objects detected for blackout (credit card and cash banknotes).
- **Right (Final Redacted Photograph)**:
  - Opaque black blackout bounding boxes are applied precisely over private items, keeping the rest of the image intact for downstream visual reasoning.

---

### Key Features

- **Task-Aware Selective Redaction**: Resolves the privacy-utility tradeoff by using question semantics to distinguish between *privacy threats* and *task-essential visual context*.
- **Multi-Model Grounding & Detection**:
  - **Florence-2 Vision-Language Grounding**: Phrase grounding and open-vocabulary localization of sensitive items and query objects.
  - **Fine-Tuned Faster R-CNN Detector**: Detects domain-specific sensitive objects (credit cards, currency, IDs, documents).
  - **OCR & Text Recognition**: Identifies sensitive textual information (names, numbers, addresses, account details).
- **Keep/Redact Overlap Logic**: Calculates Intersection-over-Union (IoU) and containment between query-critical regions and sensitive boxes to prevent accidental redaction of answered subjects.
- **Interactive Streamlit Web Dashboard**: Human-in-the-loop inspection with customizable confidence thresholds, keep-overlap ratios, blackout padding, and instant redacted image download.

---

### Dataset Details

PrivacyLens is built and evaluated on the **VizWiz-Priv** dataset:
- **Origin**: Sourced from real-world photos taken by people who are blind, paired with spoken questions.
- **Privacy Annotations**: Contains bounding boxes marking sensitive content across multiple categories (financial information, identification, personal mail, private screens, medical data, etc.).
- **Annotation Split**: Official 	rain.json and al.json parsed from data/raw/Annotations/Annotations/.
- **Smart Image Streaming**: Includes a built-in range-request streaming downloader and local cache manager (src/data/dataset_cache.py) to stream and cache only required images locally on demand.

---

### Model & Training Pipeline

#### Architectures
- **System C (Active Detector)**: Faster R-CNN with ResNet-50-FPN-v2 or MobileNetV3-Large-FPN backbone with custom anchor generation optimized for VizWiz sensitive region aspect ratios and scales.
- **System B / Baseline**: Pretrained COCO baseline with fine-tuned classification and regression heads.
- **Vision-Language Pipeline**: Microsoft Florence-2 (microsoft/Florence-2-base or microsoft/Florence-2-large) for zero-shot question grounding and open-vocabulary phrase localization.

#### Training Configuration
- **Loss Function**: Multi-task Faster R-CNN loss (Classification + Smooth L1 Bounding Box Regression).
- **Optimization**: SGD with momentum (0.9) or AdamW, cosine annealing / step learning rate schedule.
- **Evaluation Metrics**: COCO mean Average Precision (mAP@0.50, mAP@0.50:0.95), IoU tracking, and sensitive-recall score.

---

### Repository Structure

`
PrivacyLens/
│
├── docs/
│   └── images/
│       └── demo_redaction_comparison.png       # Streamlit UI demonstration screenshot
│
├── data/
│   └── raw/Annotations/                        # VizWiz-Priv annotation JSON files
│
├── notebooks/
│   └── privacylens_training.ipynb              # End-to-end training & research notebook
│
├── scripts/
│   ├── download_images.py                      # Stream / cache VizWiz image dataset
│   ├── run_baseline.py                         # Pretrained baseline evaluation
│   └── run_training.py                         # Model fine-tuning script
│
├── src/
│   ├── data/                                   # Dataset loaders, caching, torch datasets
│   ├── models/                                 # Faster R-CNN architectures & training engine
│   ├── pipeline/                               # Florence-2 grounding, OCR, and redaction logic
│   ├── evaluation/                             # COCO mAP metrics & visualization tools
│   └── ui/
│       └── app.py                              # Streamlit human-in-the-loop dashboard
│
├── tests/                                      # Unit and integration test suites
├── download_annotations.py                     # Automated annotation download script
└── requirements.txt                            # Project dependencies
`

---

### Getting Started

#### 1. Clone the Repository
`ash
git clone https://github.com/jui1ee/PrivacyLens.git
cd PrivacyLens
`

#### 2. Environment Setup
Create a virtual environment and install dependencies:

`ash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
`

#### 3. Download Annotations & Dataset
Download and extract the VizWiz-Priv annotations:

`ash
python download_annotations.py
`

To selectively download or cache images for training/evaluation:
`ash
python -m scripts.download_images
`

#### 4. Running the Interactive Streamlit UI
Launch the PrivacyLens visual comparison dashboard:

`ash
streamlit run src/ui/app.py
`
Open your browser at http://localhost:8501 to test custom image uploads, adjust detector thresholds, inspect bounding overlays, and download redacted results.

#### 5. Running Model Training & Experiments
To train the sensitive detector head:

`ash
# Train System C (Custom sensitive detector)
python -m scripts.run_training --system C

# Or run the end-to-end research notebook headlessly:
jupyter nbconvert --to notebook --execute --inplace notebooks/privacylens_training.ipynb
`
