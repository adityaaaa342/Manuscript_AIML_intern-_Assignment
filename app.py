"""Streamlit Web UI for Manuscript Layout Region Detection.

Provides an interactive interface for uploading manuscript images,
running the classical CV detection pipeline, and viewing annotated
results with bounding boxes, labels, and confidence scores.
"""

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Import the detection pipeline modules
# ---------------------------------------------------------------------------
from src.classifier import classify_regions
from src.config import CLASS_COLORS, CLASS_NAMES
from src.detector import ClassicalDetector
from src.postprocess import postprocess
from src.preprocess import preprocess_image, map_boxes_to_original
from src.visualize import draw_regions
from src.io_utils import build_image_result

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Manuscript Layout Detector",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a premium dark UI
# ---------------------------------------------------------------------------
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #e2e8f0; font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p { color: #94a3b8; font-size: 1rem; margin: 0.5rem 0 0 0; font-weight: 300; }

    .stat-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15); }
    .stat-value { font-size: 2rem; font-weight: 700; color: #818cf8; line-height: 1.2; }
    .stat-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.3rem; }

    .region-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.5px; margin: 2px 4px;
    }
    .badge-header { background: rgba(0,200,255,0.15); color: #00c8ff; border: 1px solid rgba(0,200,255,0.3); }
    .badge-footer { background: rgba(255,100,0,0.15); color: #ff8c42; border: 1px solid rgba(255,100,0,0.3); }
    .badge-main_text { background: rgba(0,255,100,0.15); color: #00ff64; border: 1px solid rgba(0,255,100,0.3); }
    .badge-side_text { background: rgba(200,0,255,0.15); color: #c864ff; border: 1px solid rgba(200,0,255,0.3); }
    .badge-filler { background: rgba(255,128,128,0.15); color: #ff8080; border: 1px solid rgba(255,128,128,0.3); }

    .info-panel {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
    }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }

    .app-footer {
        text-align: center; color: #475569; padding: 1.5rem;
        font-size: 0.8rem; border-top: 1px solid rgba(255,255,255,0.05); margin-top: 2rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
''', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

@st.cache_resource
def get_detector():
    return ClassicalDetector()


def process_image(image_bytes, filename, detector):
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return {"error": f"Could not decode image: {filename}"}

    h, w = image.shape[:2]
    t0 = time.perf_counter()

    proc_image, inv_mat = preprocess_image(image)
    detections = detector.predict(proc_image)

    if inv_mat is not None and detections:
        boxes = np.array([d["bbox"] for d in detections])
        boxes = map_boxes_to_original(boxes, inv_mat)
        for det, box in zip(detections, boxes):
            det["bbox"] = box.tolist()

    detections = classify_regions(detections, w, h, image)
    detections = postprocess(detections, w, h)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    regions = []
    for det in detections:
        regions.append({
            "label": det["label"],
            "bbox": [round(v, 1) for v in det["bbox"]],
            "confidence": round(det["confidence"], 4),
        })

    annotated = draw_regions(image, regions)
    result_json = build_image_result(filename, w, h, regions)

    return {
        "original_image": image,
        "annotated_image": annotated,
        "result_json": result_json,
        "regions": regions,
        "time_ms": elapsed_ms,
        "width": w,
        "height": h,
    }


def cv2_to_pil(cv_image):
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def create_download_zip(results):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in results:
            if "error" in r:
                continue
            fname = r["result_json"]["image"]
            stem = Path(fname).stem
            _, img_bytes = cv2.imencode(".jpg", r["annotated_image"])
            zf.writestr(f"annotated/{stem}_annotated.jpg", img_bytes.tobytes())
            json_str = json.dumps(r["result_json"], indent=2, ensure_ascii=False)
            zf.writestr(f"json/{stem}.json", json_str)
        summary = {
            "total_images": len([r for r in results if "error" not in r]),
            "results": [r["result_json"] for r in results if "error" not in r],
        }
        zf.writestr("summary.json", json.dumps(summary, indent=2, ensure_ascii=False))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔧 Settings")
    st.markdown("---")
    st.markdown("### Detection Classes")
    class_html = ""
    for cls in CLASS_NAMES:
        class_html += f'<span class="region-badge badge-{cls}">{cls}</span> '
    st.markdown(class_html, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Supported Formats")
    st.markdown('''
    <div class="info-panel">
        <p style="color: #94a3b8; margin: 0; font-size: 0.85rem;">
            JPG, JPEG, PNG, TIFF, BMP
        </p>
    </div>
    ''', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------

st.markdown('''
<div class="main-header">
    <h1>📜 Manuscript Layout Region Detector</h1>
    <p>Classical Computer Vision pipeline for detecting layout regions in palm-leaf & paper manuscripts</p>
</div>
''', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload manuscript images",
    type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
    help="Upload one or more manuscript scans. Supported: JPG, PNG, TIFF, BMP",
)

if uploaded_files:
    detector = get_detector()
    all_results = []
    progress_bar = st.progress(0, text="Processing images...")

    for idx, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress(
            (idx) / len(uploaded_files),
            text=f"Processing {uploaded_file.name} ({idx + 1}/{len(uploaded_files)})..."
        )
        result = process_image(uploaded_file.read(), uploaded_file.name, detector)
        result["filename"] = uploaded_file.name
        all_results.append(result)

    progress_bar.progress(1.0, text="All images processed!")
    time.sleep(0.5)
    progress_bar.empty()

    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]

    total_regions = sum(len(r.get("regions", [])) for r in successful)
    avg_time = sum(r.get("time_ms", 0) for r in successful) / max(len(successful), 1)
    all_confs = [reg["confidence"] for r in successful for reg in r.get("regions", [])]
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''
        <div class="stat-card">
            <div class="stat-value">{len(successful)}</div>
            <div class="stat-label">Images Processed</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'''
        <div class="stat-card">
            <div class="stat-value">{total_regions}</div>
            <div class="stat-label">Regions Detected</div>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''
        <div class="stat-card">
            <div class="stat-value">{avg_conf:.2f}</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        ''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''
        <div class="stat-card">
            <div class="stat-value">{avg_time:.0f}ms</div>
            <div class="stat-label">Avg Processing Time</div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if successful:
        zip_data = create_download_zip(successful)
        st.download_button(
            label="Download All Results (ZIP)",
            data=zip_data,
            file_name="manuscript_detection_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

    st.markdown("---")

    for i, result in enumerate(all_results):
        if "error" in result:
            st.error(f"{result.get('filename', 'Unknown')}: {result['error']}")
            continue

        fname = result["filename"]
        regions = result.get("regions", [])

        with st.expander(f"{fname}  --  {len(regions)} regions detected  |  {result['time_ms']:.0f}ms", expanded=(i == 0)):
            tab1, tab2, tab3 = st.tabs(["Annotated Result", "JSON Output", "Region Details"])

            with tab1:
                view_mode = st.radio(
                    "View:", ["Annotated", "Original", "Side by Side"],
                    horizontal=True, key=f"view_{i}",
                )
                if view_mode == "Annotated":
                    st.image(cv2_to_pil(result["annotated_image"]), caption=f"Annotated - {fname}", use_container_width=True)
                elif view_mode == "Original":
                    st.image(cv2_to_pil(result["original_image"]), caption=f"Original - {fname}", use_container_width=True)
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(cv2_to_pil(result["original_image"]), caption="Original", use_container_width=True)
                    with c2:
                        st.image(cv2_to_pil(result["annotated_image"]), caption="Annotated", use_container_width=True)

                _, img_buf = cv2.imencode(".jpg", result["annotated_image"])
                st.download_button(
                    label=f"Download Annotated Image",
                    data=img_buf.tobytes(),
                    file_name=f"{Path(fname).stem}_annotated.jpg",
                    mime="image/jpeg",
                    key=f"dl_img_{i}",
                )

            with tab2:
                json_str = json.dumps(result["result_json"], indent=2, ensure_ascii=False)
                st.code(json_str, language="json")
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{Path(fname).stem}.json",
                    mime="application/json",
                    key=f"dl_json_{i}",
                )

            with tab3:
                if regions:
                    class_counts = {}
                    for reg in regions:
                        lbl = reg["label"]
                        class_counts[lbl] = class_counts.get(lbl, 0) + 1
                    badges_html = ""
                    for cls, count in sorted(class_counts.items()):
                        badges_html += f'<span class="region-badge badge-{cls}">{cls}: {count}</span> '
                    st.markdown(badges_html, unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    table_data = []
                    for j, reg in enumerate(regions):
                        bbox = reg["bbox"]
                        table_data.append({
                            "#": j + 1,
                            "Label": reg["label"],
                            "Confidence": f"{reg['confidence']:.2f}",
                            "X1": bbox[0], "Y1": bbox[1], "X2": bbox[2], "Y2": bbox[3],
                            "Width": round(bbox[2] - bbox[0], 1),
                            "Height": round(bbox[3] - bbox[1], 1),
                        })
                    st.dataframe(table_data, use_container_width=True, hide_index=True)
                else:
                    st.info("No regions detected in this image.")

    if failed:
        st.markdown("---")
        st.markdown("### Failed Images")
        for r in failed:
            st.error(f"{r.get('filename', 'Unknown')}: {r.get('error', 'Unknown error')}")

else:
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('''
        <div class="stat-card">
            <div class="stat-value">📤</div>
            <div class="stat-label">Upload Images</div>
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">
                Drag & drop or browse for manuscript scans
            </p>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="stat-card">
            <div class="stat-value">⚡</div>
            <div class="stat-label">Instant Detection</div>
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">
                Classical CV processes each image in milliseconds
            </p>
        </div>
        ''', unsafe_allow_html=True)
    with col3:
        st.markdown('''
        <div class="stat-card">
            <div class="stat-value">📊</div>
            <div class="stat-label">Structured Output</div>
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">
                Annotated images + JSON with bounding boxes & confidence
            </p>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
    <div class="info-panel">
        <h4 style="color: #e2e8f0; margin-top: 0;">Detected Layout Regions</h4>
        <p style="color: #94a3b8; margin: 0.3rem 0;">
            The system identifies <strong>5 region types</strong> in manuscript scans:
        </p>
        <ul style="color: #94a3b8; margin: 0.5rem 0;">
            <li><strong style="color: #00c8ff;">header</strong> - Running headers, section titles, folio numbers</li>
            <li><strong style="color: #ff8c42;">footer</strong> - Catchwords, page numbers, signatures</li>
            <li><strong style="color: #00ff64;">main_text</strong> - Primary manuscript body text</li>
            <li><strong style="color: #c864ff;">side_text</strong> - Marginalia, annotations, commentary</li>
            <li><strong style="color: #ff8080;">filler</strong> - Decorative elements, English metadata, watermarks</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)

st.markdown('''
<div class="app-footer">
    Manuscript Layout Region Detection System  |  Classical Computer Vision Pipeline  |  Built with Streamlit
</div>
''', unsafe_allow_html=True)
