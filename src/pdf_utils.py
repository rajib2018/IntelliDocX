from __future__ import annotations

import io
from typing import List

from PIL import Image

# PyMuPDF is installed as "PyMuPDF" but imported as "fitz" (legacy name).
import fitz  # type: ignore


def pdf_bytes_to_images(pdf_bytes: bytes, max_pages: int = 10, dpi: int = 200) -> List[Image.Image]:
    """
    Render a PDF (bytes) into a list of PIL Images.

    Notes:
    - Streamlit file_uploader provides bytes; PyMuPDF can open from bytes using a memory stream.
    - dpi controls rasterization quality (and speed).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images: List[Image.Image] = []

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i in range(min(max_pages, doc.page_count)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)

    doc.close()
    return images
