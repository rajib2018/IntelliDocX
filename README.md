# IDP OCR (Streamlit) – RapidOCR (ONNX) without pytesseract

This repository is a minimal **OCR + Intelligent Document Processing (IDP)** starter kit that you can deploy on:
- GitHub (source control)
- Streamlit Community Cloud (1-click deploy)

**Key points**
- OCR is performed with **RapidOCR (ONNX Runtime)** (no `pytesseract`).
- PDFs are rendered to images with **PyMuPDF** (installed as `PyMuPDF`, imported as `fitz`).
- “IDP” layer includes a lightweight heuristic **document type detector** and **field extraction** (regex + normalization).
- Results can be downloaded as **JSON** and **text**.

## Project structure

```
.
├── app.py
├── requirements.txt
├── runtime.txt
├── src/
│   ├── ocr_engine.py
│   ├── pdf_utils.py
│   └── extraction.py
└── tests/
    └── test_extraction.py
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit deployment notes

- Put `requirements.txt` and `runtime.txt` in the repo root.
- If you see `Unable to import fitz`, do **not** add the outdated `fitz` PyPI package. Install `PyMuPDF` and import `fitz`.  
  This “fitz vs PyMuPDF” confusion is a common Streamlit deployment issue.

## Extending to “real” IDP

This starter is intentionally lightweight. Common production upgrades:
- Replace heuristic doc-type detection with a trained classifier (TF‑IDF + LR/SVM) or LLM-based classification.
- Add layout-aware parsing (tables/line-items) and key-value extraction models.
- Add human-in-the-loop review workflow and confidence thresholds.
- Persist outputs (S3 / Postgres / vector DB) and integrate into downstream systems.

## License

MIT (see `LICENSE`).
