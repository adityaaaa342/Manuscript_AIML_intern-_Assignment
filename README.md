# 📜 Manuscript Layout Region Detection

> **Automated layout analysis and region classification for historical Indic manuscripts (palm-leaf, birch bark, and paper folios) using Classical Computer Vision and Deep Learning.**

[![Live App](https://img.shields.io/badge/🌐_Live_Demo-Render_Web_Service-brightgreen.svg)](#-live-demo)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Pytest-23%20Passed-brightgreen.svg)](https://docs.pytest.org/)

---

## 🌐 Live Demo

You can try the interactive manuscript detection Web UI directly in your browser without installing anything locally:

👉 **[Launch Live Web App on Render](https://manuscript-layout-detector.onrender.com)** *(Replace with your Render deployment URL)*

---

## 📌 Table of Contents
- [Live Demo](#-live-demo)
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Target Classes](#-target-classes)
- [How to Run Locally](#-how-to-run-locally-optional)
- [CLI Batch Inference](#-cli-batch-inference)
- [Project Structure](#-project-structure)
- [Testing & Validation](#-testing--validation)
- [Deploying on Render](#-deploying-on-render)

---

## 🌟 Overview

Historical manuscripts (like palm-leaf and handmade paper folios) often suffer from aging, uneven lighting, dark background borders, and digital watermarks.

This system automatically detects, segments, and classifies layout regions across manuscript scans into **5 standardized categories**:
- **Line-by-line body text** with straight, consistent text margins.
- **Folio headers & footers** (titles, running numbers, colophons).
- **Marginalia / side annotations**.
- **Digital metadata stamps, library banners, and decorative filler**.

---

## 🚀 Key Features

- **⚡ Line-by-Line Segmentation**: Accurately detects each text line individually without merging lines into giant blocks or causing jagged borders.
- **🎨 Interactive Web UI**: Built with Streamlit for drag-and-drop batch image uploads, visual inspection, side-by-side comparisons, and one-click ZIP downloads.
- **🧠 Dual-Engine Architecture**:
  - **Classical CV Engine**: Runs out-of-the-box on any standard CPU using adaptive thresholding, leaf contour detection, and projection profiling (0 training data required).
  - **YOLO Engine**: Extensible YOLO deep learning pipeline (`train.py`) ready for fine-tuning on custom labeled datasets.
- **📦 Structured JSON Export**: Generates standardized JSON bounding box metadata (`[x1, y1, x2, y2]`, confidence score, label) for every image.
- **✅ Robust Test Suite**: 23 automated unit tests covering batch processing, schema validation, corrupted inputs, and post-processing.

---

## 🏷️ Target Classes

| Class | Color | Description |
| :--- | :---: | :--- |
| `header` | 🟧 Orange | Running headers, folio titles, chapter names at the top margin |
| `main_text` | 🟩 Green | Individual text lines of the primary manuscript body |
| `footer` | 🟦 Blue | Signatures, catchwords, bottom margin metadata, page numbers |
| `side_text` | 🟪 Magenta | Marginalia, scribe commentary, notes along left/right margins |
| `filler` | 🟥 Salmon | Digital library stamps (e.g. eGangotri/Sringeri banners), watermarks, non-target script |

---

## 💻 How to Run Locally (Optional)

Running locally is completely optional. If you want to develop or test offline on your own computer:

### 1. Clone the Repository
```bash
git clone https://github.com/adityaaaa342/Manuscript_AIML_intern-_Assignment.git
cd Manuscript_AIML_intern-_Assignment
```

### 2. Create and Activate a Virtual Environment
- **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies & Launch
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚙️ CLI Batch Inference

Run batch processing directly from the command line:

```bash
# Process all images in data/test_images and save to results/
python inference.py --input ./data/test_images --output ./results
```

### Output Directory Structure
```
results/
├── Sample_img_1_annotated.jpg   # Visual overlay with bounding boxes
├── Sample_img_1.json            # Bounding box coordinates & classes
├── Sample_img_02_annotated.jpg
├── Sample_img_02.json
└── summary.json                 # Batch execution summary
```

---

## 📂 Project Structure

```
manuscript-layout/
├── app.py                     # Streamlit Web UI Application
├── inference.py               # CLI entry point for batch inference
├── train.py                   # YOLO fine-tuning pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── data/
│   └── test_images/           # Sample manuscript scans
├── results/                   # Output folder for JSON & annotated images
├── src/
│   ├── __init__.py
│   ├── config.py              # Class names, colors, thresholds
│   ├── detector.py            # Classical line-level detector & YOLO wrapper
│   ├── classifier.py          # Spatial & heuristic classification rules
│   ├── preprocess.py          # Illumination normalization & deskewing
│   ├── postprocess.py         # NMS, clipping, small box filtering
│   ├── visualize.py           # Bounding box rendering utilities
│   └── io_utils.py            # Safe image loading and JSON serialization
└── tests/
    ├── conftest.py            # Test fixtures & synthetic image generators
    ├── test_batch.py          # Batch processing tests
    ├── test_corrupt.py        # Corrupted/invalid image tests
    ├── test_postprocess.py    # Box clipping & NMS validation
    └── test_schema.py         # JSON schema compliance tests
```

---

## 🧪 Testing & Validation

Run the complete test suite with `pytest`:

```bash
pytest tests/
```

---

## ☁️ Deploying on Render

To deploy the Web UI live on [Render](https://render.com/):

1. Log in to [Render Dashboard](https://dashboard.render.com/) and click **New +** $\rightarrow$ **Web Service**.
2. Connect your GitHub repository.
3. Configure the service settings:
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false --server.enableXsrfProtection false
     ```
   - **Plan**: `Free`
4. Click **Deploy Web Service**. Render will provide your public live URL.

---

## 📄 License
This project is open-source and available under the MIT License.
