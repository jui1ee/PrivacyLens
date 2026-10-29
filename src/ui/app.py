import os
import io
import sys
from pathlib import Path
import streamlit as st
from PIL import Image

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
import src.pipeline.florence_redact_pipeline as frp
importlib.reload(frp)
from src.pipeline.florence_redact_pipeline import run_question_aware_redaction

st.set_page_config(
    page_title="PrivacyLens: Question-Aware Redaction",
    page_icon="🛡️",
    layout="wide"
)

# Custom Aesthetic Theme
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #8b949e;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛡️ PrivacyLens: Question-Aware Redaction Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Florence-2 OCR & phrase grounding + trained Faster R-CNN sensitive detector</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("⚙️ Pipeline Configuration")
florence_version = st.sidebar.selectbox("Florence-2 Model", ["microsoft/Florence-2-base", "microsoft/Florence-2-large"], index=0)

detector_options = {
    "v2 Active (System C - Peak 87.3% mAP)": "models/checkpoints/privacylens_best_c.pt",
    "v1 Baseline (323 images)": "models/checkpoints/privacylens_best.pt",
    "None (Pure Florence-2 & Regex)": ""
}
detector_selection = st.sidebar.selectbox("Sensitive Detector Checkpoint", list(detector_options.keys()), index=0)
detector_path = detector_options[detector_selection]

keep_overlap_thresh = st.sidebar.slider("Keep Overlap Threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
                                       help="Sensitive region is kept if >= threshold overlaps with question-needed region")
detector_score_thresh = st.sidebar.slider("Detector Confidence Threshold", min_value=0.10, max_value=0.80, value=0.25, step=0.05,
                                         help="Score threshold for Faster R-CNN sensitive detector")
enable_visual_grounding = st.sidebar.checkbox("Visual Sensitive Grounding (Cards, Cash, IDs)", value=True,
                                             help="Uses Florence-2 phrase grounding to detect physical cards, cash, and documents even when text is degraded")
box_padding = st.sidebar.slider("Blackout Box Padding (px)", min_value=0, max_value=15, value=4, step=1)

# Main layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Upload Photograph & Query")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])
    user_query = st.text_input(
        "Question / Intended Task",
        placeholder="e.g., why is my file not downloading / what are the ingredients?"
    )

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Original Photograph", use_container_width=True)

with col_right:
    st.subheader("2. Run Question-Aware Redaction")
    if uploaded_file is not None:
        if st.button("🚀 Process & Redact", type="primary", use_container_width=True):
            with st.spinner("Executing Florence-2 OCR, Phrase Grounding, and Sensitive Redaction..."):
                try:
                    redacted_img, overlay_img, report = run_question_aware_redaction(
                        image=input_image,
                        question=user_query,
                        detector_ckpt=detector_path,
                        florence_model_id=florence_version,
                        keep_overlap=keep_overlap_thresh,
                        padding=box_padding,
                        detector_score_thresh=detector_score_thresh,
                        enable_visual_grounding=enable_visual_grounding
                    )
                    st.session_state["florence_result"] = {
                        "redacted_img": redacted_img,
                        "overlay_img": overlay_img,
                        "report": report
                    }
                except Exception as e:
                    st.error(f"Error executing pipeline: {e}")

if "florence_result" in st.session_state and uploaded_file is not None:
    res = st.session_state["florence_result"]
    report = res["report"]

    st.markdown("---")
    st.subheader("3. Redaction Decisions & Conflict Report")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("OCR Lines Detected", report["all_ocr_lines"])
    m2.metric("Question-Needed Regions", report["needed_boxes_count"])
    m3.metric("Redacted (Blacked Out)", len(report["redacted"]))
    m4.metric("Kept (Needed for Query)", len(report["kept_needed"]))

    def format_kind(kind: str) -> str:
        if "credit card" in kind or "debit card" in kind:
            return "💳 Credit Card"
        if "money" in kind or "dollar" in kind or "banknote" in kind:
            return "💵 Currency / Banknote"
        if "pii_text" in kind:
            return "🔤 PII Text"
        if "detector" in kind:
            return "🎯 Trained Detector (Private Visual Region)"
        return "🛡️ " + kind

    with st.expander("📋 Detailed Inspection & Conflict Summary", expanded=True):
        if report["kept_needed"]:
            st.warning("⚠️ **Conflict Handled:** The following sensitive boxes overlap with the region needed to answer the question and were preserved:")
            for item in report["kept_needed"]:
                text_info = f" | **Text:** `'{item['text']}'`" if item['text'] else ""
                st.write(f"- **Box:** `{item['box']}` | **Type:** `{format_kind(item['kind'])}`{text_info} (Overlap: {item['overlap_with_needed']*100:.0f}%)")
        else:
            st.success("✅ No sensitive regions overlapped with question-needed content.")

        if report["redacted"]:
            st.info("⬛ **Redacted Regions:**")
            for item in report["redacted"]:
                text_info = f" | **Text:** `'{item['text']}'`" if item['text'] else ""
                st.write(f"- **Box:** `{item['box']}` | **Type:** `{format_kind(item['kind'])}`{text_info}")

    st.markdown("---")
    st.subheader("4. Visual Comparison & Download")
    col_vis, col_out = st.columns(2)
    with col_vis:
        st.image(res["overlay_img"], caption="Grounding Overlay (Green=Needed for Question, Red=Redacted, Yellow=Kept Needed)", use_container_width=True)
    with col_out:
        st.image(res["redacted_img"], caption="Final Redacted Photograph (Opaque Black)", use_container_width=True)

        buf = io.BytesIO()
        res["redacted_img"].save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="💾 Download Redacted Image",
            data=byte_im,
            file_name="redacted_output.png",
            mime="image/png",
            type="primary",
            use_container_width=True
        )
