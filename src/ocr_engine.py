from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import cv2  # type: ignore

# RapidOCR (ONNX runtime) - no pytesseract required.
from rapidocr_onnxruntime import RapidOCR  # type: ignore


@dataclass
class OcrLine:
    box: List[List[int]]  # 4 points: [[x,y],...]
    text: str
    score: float


def get_ocr_engine() -> RapidOCR:
    """
    Create the OCR engine once. Streamlit should cache this via st.cache_resource.
    """
    # RapidOCR has multiple optional params; defaults are fine for most documents.
    return RapidOCR()


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _preprocess(bgr: np.ndarray, deskew: bool) -> np.ndarray:
    """
    Basic preprocessing to improve OCR quality:
    - grayscale
    - denoise
    - adaptive threshold
    - optional deskew
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold tends to help on uneven lighting
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )

    if not deskew:
        return th

    # Deskew via minAreaRect on foreground pixels
    coords = np.column_stack(np.where(th < 255))
    if coords.size == 0:
        return th

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = th.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(th, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def _draw_boxes(base_img: Image.Image, lines: List[OcrLine]) -> bytes:
    """
    Return a PNG bytes overlay with OCR boxes.
    """
    img = base_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    for ln in lines:
        pts = [(int(x), int(y)) for x, y in ln.box]
        # draw polygon
        draw.line(pts + [pts[0]], width=2)
        # label near first point
        if ln.text:
            x0, y0 = pts[0]
            draw.text((x0 + 2, y0 + 2), ln.text[:40], fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ocr_image(
    engine: RapidOCR,
    image: Image.Image,
    do_preprocess: bool = True,
    deskew: bool = False,
    return_word_boxes: bool = True,
) -> Dict[str, Any]:
    """
    Run OCR on a PIL image and return:
      - text: full text joined with newlines
      - lines: [{box,text,score}, ...]
      - visualization_png: optional PNG bytes with OCR boxes drawn
    """
    bgr = _pil_to_bgr(image)
    inp = bgr
    if do_preprocess:
        inp = _preprocess(bgr, deskew=deskew)

    # RapidOCR accepts numpy arrays.
    out = engine(inp)

    # Output shapes can differ by version:
    #   (result, elapse) OR result
    if isinstance(out, tuple) and len(out) >= 1:
        result = out[0]
    else:
        result = out

    lines: List[OcrLine] = []
    if result:
        for item in result:
            # item: [box, text, score]
            try:
                box, txt, score = item
                lines.append(OcrLine(box=box, text=str(txt), score=float(score)))
            except Exception:
                continue

    # Build text (best effort). Keep order returned by engine.
    text = "\n".join([ln.text for ln in lines if ln.text])

    payload: Dict[str, Any] = {
        "text": text,
        "lines": [ln.__dict__ for ln in lines],
    }

    # Visualization
    if return_word_boxes and lines:
        payload["visualization_png"] = _draw_boxes(image, lines)

    return payload
