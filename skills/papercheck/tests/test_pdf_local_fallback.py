import importlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RULES_ROOT = ROOT / "assets" / "paperchecker-rules"


def load_pdf_extractor(monkeypatch):
    monkeypatch.syspath_prepend(str(RULES_ROOT))
    sys.modules.pop("core.extractor.pdf_extractor", None)
    module = importlib.import_module("core.extractor.pdf_extractor")
    return module.PDFExtractor


def test_local_pdf_prefers_pymupdf4llm_markdown(monkeypatch):
    PDFExtractor = load_pdf_extractor(monkeypatch)
    fake = types.SimpleNamespace(
        to_markdown=lambda path: "# Title\n\nBody text from pymupdf4llm with enough content."
    )
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake)

    extractor = PDFExtractor()
    text = extractor._extract_pdf_locally("paper.pdf")

    assert "Body text from pymupdf4llm" in text
    assert extractor._last_pdf_extraction_method == "pymupdf4llm"


def test_local_pdf_falls_back_to_fitz_when_markdown_empty(monkeypatch):
    PDFExtractor = load_pdf_extractor(monkeypatch)
    monkeypatch.setitem(sys.modules, "pymupdf4llm", types.SimpleNamespace(to_markdown=lambda path: ""))

    class FakePage:
        def get_text(self, mode="text", sort=False):
            return "Page text from fitz with enough characters for fallback."

    class FakeDoc:
        def __len__(self):
            return 1

        def load_page(self, page_num):
            return FakePage()

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "fitz", types.SimpleNamespace(open=lambda path: FakeDoc()))

    extractor = PDFExtractor()
    text = extractor._extract_pdf_locally("paper.pdf")

    assert "Page text from fitz" in text
    assert "<!-- page 1 -->" in text
    assert extractor._last_pdf_extraction_method == "pymupdf_fitz"


def test_local_pdf_reports_install_action_when_local_extractors_missing(monkeypatch):
    PDFExtractor = load_pdf_extractor(monkeypatch)
    monkeypatch.setitem(sys.modules, "pymupdf4llm", None)
    monkeypatch.setitem(sys.modules, "fitz", None)

    extractor = PDFExtractor()
    try:
        extractor._extract_pdf_locally("paper.pdf")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing local PDF dependencies to raise RuntimeError")

    assert "Install PyMuPDF and pymupdf4llm" in message
    assert "configure MINERU_API_KEY" in message
