from src.extraction import detect_document_type, extract_fields


def test_detect_invoice():
    text = "TAX INVOICE\nInvoice No: INV-12345\nAmount Due: USD 1,234.56\nInvoice Date: 12/11/2025"
    assert detect_document_type(text) == "invoice"


def test_extract_invoice_fields():
    text = "Invoice No: INV-12345\nAmount Due: USD 1,234.56\nInvoice Date: 12/11/2025\nVAT: USD 12.34"
    out = extract_fields(text, doc_type="invoice", custom_rules={})
    assert out.get("invoice_number") == "INV-12345"
    assert abs(out.get("total_amount") - 1234.56) < 1e-6
    assert out.get("invoice_date") in ("2025-11-12", "2025-12-11")  # dayfirst parsing
    assert abs(out.get("tax_amount") - 12.34) < 1e-6
