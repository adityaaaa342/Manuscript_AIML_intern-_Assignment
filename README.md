# 📜 Manuscript Layout Region Detection

> **Automated layout analysis and region classification for historical Indic manuscripts (palm-leaf, birch bark, and paper folios) using Classical Computer Vision and Deep Learning.**

[![Live App](https://img.shields.io/badge/🌐_Live_Demo-Hosted_Web_Service-brightgreen.svg)](https://manuscript-aiml-intern-assignment.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8.svg)](https://opencv.org/)
[![Tests](https://img.shields.io/badge/Pytest-23%20Passed-brightgreen.svg)](https://docs.pytest.org/)

---

## 🎯 How to Run the Application (Choose One)

You can run and test this project using either of the two approaches below:

### 🟢 Approach 1: Run Online via Hosted Web App (Recommended)
**No installation, setup, or coding required.** Test the model directly in your web browser:

👉 **[Launch Live Manuscript Detector Web App](https://manuscript-aiml-intern-assignment.onrender.com)**

1. Open the hosted URL in any browser (desktop or mobile).
2. Drag and drop manuscript scans (`.jpg`, `.png`, `.tiff`, `.bmp`).
3. View real-time visual bounding boxes, confidence scores, and structured JSON results.
4. Download annotated images or batch JSON results with one click.

---

### 💻 Approach 2: Run Manually on Your Local Machine

If you are a developer and want to run, modify, or test the code offline:

#### 1. Clone the Repository
```bash
git clone https://github.com/adityaaaa342/Manuscript_AIML_intern-_Assignment.git
cd Manuscript_AIML_intern-_Assignment
```

#### 2. Create and Activate a Virtual Environment
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

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Launch the Interactive Web UI Locally
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

#### 5. Or Run Batch Processing from the Command Line (CLI)
```bash
python inference.py --input ./data/test_images --output ./results
```

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

## 📂 Project Structure

```
manuscript-layout/
├── 🌐 app.py                 # Interactive Web UI (Streamlit)
├── ⚡ inference.py           # Run detection on multiple images (CLI)
├── 🧠 train.py               # Model training pipeline (YOLO)
│
├── 📁 src/                   # Core Detection Pipeline
│   ├── detector.py           # Detects text lines & layout regions
│   ├── preprocess.py         # Cleans scans & normalizes lighting
│   ├── postprocess.py        # Refines & filters bounding boxes
│   ├── visualize.py          # Draws color-coded boxes on images
│   ├── config.py             # Target classes, colors, and settings
│   └── io_utils.py           # Image loader & JSON file exporter
│
├── 📁 data/test_images/      # Input folder for manuscript scans
├── 📁 results/               # Output folder for JSON & annotated images
└── 📁 tests/                 # Automated test suite (23 unit tests)
```

---

## 🧪 Testing & Validation

Run the complete test suite with `pytest`:

```bash
pytest tests/
```

**Test Coverage Highlights:**
- Schema compliance against expected layout JSON specification.
- Handling of corrupted, empty, or unreadable image files.
- Coordinate clipping and Non-Maximum Suppression (NMS).
- Batch folder discovery and summary generation.
