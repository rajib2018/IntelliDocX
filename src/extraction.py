from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from dateutil import parser as dateparser  # type: ignore


DOC_TYPES = ["invoice", "receipt", "purchase_order", "contract", "unknown"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def detect_document_type(text: str) -> str:
    """
    Lightweight heuristic classifier.
    For production, you can swap this with a trained text classifier or LLM.
    """
    t = _norm(text)
    scores = {k: 0 for k in DOC_TYPES}

    # invoice
    for kw in ["invoice", "inv#", "amount due", "bill to", "tax invoice", "vat"]:
        if kw in t:
            scores["invoice"] += 2

    # receipt
    for kw in ["receipt", "thank you", "change", "cashier", "subtotal"]:
        if kw in t:
            scores["receipt"] += 2

    # purchase order
    for kw in ["purchase order", "po number", "ship to", "deliver to", "vendor"]:
        if kw in t:
            scores["purchase_order"] += 2

    # contract
    for kw in ["agreement", "party", "terms and conditions", "hereinafter", "whereas"]:
        if kw in t:
            scores["contract"] += 2

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def _first_match(patterns: List[str], text: str, group: int = 1, flags: int = re.IGNORECASE) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags=flags)
        if m:
            try:
                return m.group(group).strip()
            except Exception:
                return m.group(0).strip()
    return None


def _extract_amount(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    s = raw.strip()
    # Remove currency symbols/letters; keep digits , .
    s = re.sub(r"[^\d,\.]", "", s)
    if not s:
        return None
    # If both comma and dot exist, assume comma thousands: 1,234.56
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        # If only comma exists, treat as decimal in some locales, but it's ambiguous.
        # Prefer thousands separator removal when there are multiple commas.
        if s.count(",") >= 2:
            s = s.replace(",", "")
        elif s.count(",") == 1 and "." not in s:
            # assume comma decimal
            s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def _extract_date(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    try:
        dt = dateparser.parse(raw, fuzzy=True, dayfirst=True)
        return dt.date().isoformat()
    except Exception:
        return None


def extract_fields(text: str, doc_type: str = "unknown", custom_rules: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Best-effort field extraction using regex + heuristics.
    Returns normalized values; does not guarantee correctness.
    """
    custom_rules = custom_rules or {}
    t = text or ""

    out: Dict[str, Any] = {"doc_type": doc_type}

    # Common fields
    out["emails"] = list({e.lower() for e in re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", t, flags=re.I)})
    out["phones"] = list({p for p in re.findall(r"(?:\+\d{1,3}[\s-]?)?\d{6,14}", t)})

    # Generic dates
    date_raw = _first_match(
        [
            r"\bdate\s*[:\-]?\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})\b",
            r"\bdate\s*[:\-]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{2,4})\b",
        ],
        t,
        group=1,
    )
    out["date"] = _extract_date(date_raw)

    # Doc-specific
    if doc_type == "invoice":
        inv_no = _first_match(
            [
                r"\binvoice\s*(?:no|#|number)\s*[:\-]?\s*([A-Z0-9\-\/]{4,})\b",
                r"\binv\s*(?:no|#|number)?\s*[:\-]?\s*([A-Z0-9\-\/]{4,})\b",
            ],
            t,
            group=1,
        )
        out["invoice_number"] = inv_no

        inv_date = _first_match(
            [
                r"\binvoice\s*date\s*[:\-]?\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})\b",
                r"\binvoice\s*date\s*[:\-]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{2,4})\b",
            ],
            t,
            group=1,
        )
        out["invoice_date"] = _extract_date(inv_date)

        due_date = _first_match(
            [
                r"\bdue\s*date\s*[:\-]?\s*([0-9]{1,2}[\/\-.][0-9]{1,2}[\/\-.][0-9]{2,4})\b",
                r"\bdue\s*date\s*[:\-]?\s*([0-9]{1,2}\s+[A-Za-z]{3,9}\s+[0-9]{2,4})\b",
            ],
            t,
            group=1,
        )
        out["due_date"] = _extract_date(due_date)

        currency = _first_match([r"\b(usd|eur|gbp|inr|aed|sar|bhd|qar|omr|jod)\b"], t, group=1)
        out["currency"] = currency.upper() if currency else None

        total_raw = _first_match(
            [
                r"\b(total\s*(?:amount)?|amount\s*due)\s*[:\-]?\s*([A-Z]{0,3}\s?\d[\d,]*\.?\d{0,2})\b",
                r"\bgrand\s*total\s*[:\-]?\s*([A-Z]{0,3}\s?\d[\d,]*\.?\d{0,2})\b",
            ],
            t,
            group=2,
        ) or _first_match([r"\bgrand\s*total\s*[:\-]?\s*([A-Z]{0,3}\s?\d[\d,]*\.?\d{0,2})\b"], t, group=1)

        out["total_amount"] = _extract_amount(total_raw)

        tax_raw = _first_match(
            [
                r"\b(vat|tax)\s*[:\-]?\s*([A-Z]{0,3}\s?\d[\d,]*\.?\d{0,2})\b",
            ],
            t,
            group=2,
        )
        out["tax_amount"] = _extract_amount(tax_raw)

    elif doc_type == "purchase_order":
        po_no = _first_match(
            [
                r"\bpo\s*(?:no|#|number)\s*[:\-]?\s*([A-Z0-9\-\/]{4,})\b",
                r"\bpurchase\s*order\s*(?:no|#|number)\s*[:\-]?\s*([A-Z0-9\-\/]{4,})\b",
            ],
            t,
            group=1,
        )
        out["po_number"] = po_no

    # Apply custom rules (override/extend)
    custom_out: Dict[str, Any] = {}
    for field, patterns in custom_rules.items():
        # If a pattern has multiple groups, prefer last group; we attempt group 1 then 2 then 3
        val = (
            _first_match(patterns, t, group=3)
            or _first_match(patterns, t, group=2)
            or _first_match(patterns, t, group=1)
            or _first_match(patterns, t, group=0)
        )
        if val is not None:
            custom_out[field] = val
    if custom_out:
        out["custom"] = custom_out

    # clean empties
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}
