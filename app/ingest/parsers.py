from abc import ABC, abstractmethod

from app.config import settings


class PDFParser(ABC):
    """Strategy — interfaz para parsear PDFs a markdown."""

    @abstractmethod
    def parse(self, pdf_bytes: bytes) -> str:
        ...


class DoclingParser(PDFParser):
    _converter = None

    def _get_converter(self):
        if DoclingParser._converter is None:
            from docling.document_converter import DocumentConverter
            DoclingParser._converter = DocumentConverter()
        return DoclingParser._converter

    def parse(self, pdf_bytes: bytes) -> str:
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(pdf_bytes)
            tmp_path = f.name
        try:
            result = self._get_converter().convert(tmp_path)
            return result.document.export_to_markdown()
        finally:
            os.unlink(tmp_path)


_PARSERS: dict[str, type[PDFParser]] = {
    "docling": DoclingParser,
}


def get_parser(name: str | None = None) -> PDFParser:
    key = (name or settings.pdf_parser).lower()
    cls = _PARSERS.get(key)
    if cls is None:
        raise ValueError(f"Unknown PDF parser: '{key}'. Options: {list(_PARSERS)}")
    return cls()