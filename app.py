import io
import json
import traceback
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
from PIL import Image

from src.pdf_utils import pdf_bytes_to_images
from src.ocr_engine import get_ocr_engine, ocr_image
from src.extraction import detect_document_type, extract_fields

st.set_page_config(page_title="IDP OCR (RapidOCR) – Streamlit", layout="wide")

st.title("Intelligent Document Processing (IDP) – OCR + Field Extraction")
st.caption("OCR engine: RapidOCR (ONNX) • PDF rendering: PyMuPDF • No pytesseract.")

with st.sidebar:
    st.header("Settings")

    max_pages = st.number_input("Max PDF pages to process", min_value=1, max_value=50, value=10, step=1)
    dpi = st.slider("PDF render DPI (higher = slower, better OCR)", min_value=100, max_value=350, value=200, step=25)

    st.subheader("Pre-processing")
    do_preprocess = st.checkbox("Enable image pre-processing (recommended)", value=True)
    deskew = st.checkbox("Deskew (slow; helps scanned docs)", value=False)

    st.subheader("Extraction")
    enable_rules = st.checkbox("Enable heuristic field extraction", value=True)
    custom_rules_str = st.text_area(
        "Custom regex rules (JSON)",
        value=json.dumps(
            {
                "invoice_number": [r"\b(invoice|inv)\s*(no|#|number)\s*[:\-]?\s*([A-Z0-9\-\/]{4,})\b"],
                "total_amount": [r"\b(total\s*(amount)?|amount\s*due)\s*[:\-]?\s*([A-Z]{0,3}\s?\d[\d,]*\.?\d{0,2})\b"],
            },
            indent=2,
        ),
        height=180,
        help="Optional. Provide patterns to extract custom fields. Each field maps to a list of regex strings.",
    )

    st.divider()
    st.subheader("Outputs")
    draw_boxes = st.checkbox("Visualize OCR boxes", value=True)

uploaded = st.file_uploader("Upload a PDF, PNG, JPG, or JPEG", type=["pdf", "png", "jpg", "jpeg"])

@st.cache_resource
def _engine():
    # Construct once per session
    return get_ocr_engine()

def _load_custom_rules() -> Dict[str, List[str]]:
    if not custom_rules_str.strip():
        return {}
    try:
        rules = json.loads(custom_rules_str)
        if not isinstance(rules, dict):
            return {}
        # normalize: ensure list[str]
        out: Dict[str, List[str]] = {}
        for k, v in rules.items():
            if isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v):
                out[k] = v
        return out
    except Exception:
        return {}

if not uploaded:
    st.info("Upload a document to begin.")
    st.stop()

file_bytes = uploaded.read()
filename = uploaded.name.lower()

try:
    engine = _engine()

    images: List[Image.Image]
    if filename.endswith(".pdf"):
        images = pdf_bytes_to_images(file_bytes, max_pages=int(max_pages), dpi=int(dpi))
    else:
        images = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]

    st.success(f"Loaded {len(images)} page(s).")

    all_pages_results: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

    for idx, img in enumerate(images, start=1):
        st.subheader(f"Page {idx}")

        with st.spinner("Running OCR..."):
            ocr_res = ocr_image(
                engine,
                img,
                do_preprocess=do_preprocess,
                deskew=deskew,
                return_word_boxes=True,
            )

        page_text = ocr_res["text"]
        full_text_parts.append(page_text)

        cols = st.columns([1, 1])
        with cols[0]:
            st.image(img, caption=f"Original (Page {idx})", use_container_width=True)

        with cols[1]:
            if draw_boxes and ocr_res.get("visualization_png"):
                st.image(ocr_res["visualization_png"], caption="OCR boxes", use_container_width=True)
            st.text_area("OCR text", value=page_text, height=220)

        # Document intelligence on first page (or per page)
        page_doc_type = detect_document_type(page_text)
        extracted = {}
        if enable_rules:
            extracted = extract_fields(page_text, doc_type=page_doc_type, custom_rules=_load_custom_rules())

        all_pages_results.append(
            {
                "page": idx,
                "doc_type": page_doc_type,
                "text": page_text,
                "lines": ocr_res.get("lines", []),
                "extracted": extracted,
            }
        )

        if enable_rules and extracted:
            st.markdown("**Extracted fields (heuristic):**")
            st.json(extracted)

    full_text = "\n\n".join(full_text_parts)
    overall_doc_type = detect_document_type(full_text)

    st.divider()
    st.header("Document Summary")

    left, right = st.columns([1, 1])
    with left:
        st.metric("Detected document type", overall_doc_type)
        if enable_rules:
            overall_extracted = extract_fields(full_text, doc_type=overall_doc_type, custom_rules=_load_custom_rules())
            st.markdown("**Aggregated extracted fields (best effort):**")
            st.json(overall_extracted)
    with right:
        st.text_area("Full text (all pages)", value=full_text, height=260)

    # Downloads
    st.divider()
    st.header("Download results")

    result_payload = {
        "file_name": uploaded.name,
        "doc_type": overall_doc_type,
        "pages": all_pages_results,
    }

    st.download_button(
        "Download JSON",
        data=json.dumps(result_payload, indent=2).encode("utf-8"),
        file_name=f"{uploaded.name}.idp.json",
        mime="application/json",
    )
    st.download_button(
        "Download text",
        data=full_text.encode("utf-8"),
        file_name=f"{uploaded.name}.txt",
        mime="text/plain",
    )

except Exception as e:
    st.error("Processing failed.")
    st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
