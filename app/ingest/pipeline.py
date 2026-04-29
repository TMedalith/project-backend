import asyncio
import hashlib
import logging
import re

import httpx

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document

from app.ingest.parsers import PDFParser, get_parser
from app.storage.base import VectorRepository

logger = logging.getLogger(__name__)

_DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s,;\"\'<>]+)', re.IGNORECASE)


def _extract_doi(text: str) -> str | None:
    m = _DOI_RE.search(text)
    return m.group(1).rstrip(".)") if m else None


def _fetch_crossref(doi: str) -> dict:
    try:
        url = f"https://api.crossref.org/works/{doi}"
        r = httpx.get(url, timeout=8, headers={"User-Agent": "Raccly/1.0"})
        if r.status_code != 200:
            return {}
        work = r.json().get("message", {})
        authors = [
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in work.get("author", [])
        ]
        year = None
        date_parts = (
            work.get("published-print") or work.get("published-online") or {}
        ).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        journal = None
        container = work.get("container-title", [])
        if container:
            journal = container[0]
        return {
            "title":   work.get("title", [None])[0],
            "authors": authors,
            "year":    year,
            "journal": journal,
        }
    except Exception:
        logger.warning("CrossRef lookup failed for DOI %s", doi)
        return {}


def _build_metadata(filename: str, pdf_bytes: bytes, text: str) -> dict:
    source = hashlib.md5(pdf_bytes).hexdigest()
    doi = _extract_doi(text)
    crossref = _fetch_crossref(doi) if doi else {}
    return {
        "source":  source,
        "title":   crossref.get("title") or filename.removesuffix(".pdf"),
        "authors": crossref.get("authors") or [],
        "year":    crossref.get("year"),
        "doi":     doi,
        "journal": crossref.get("journal"),
    }


def ingest(
    pdf_bytes: bytes,
    filename: str,
    storage: VectorRepository,
    parser: PDFParser | None = None,
) -> dict:
    parser = parser or get_parser()
    markdown = parser.parse(pdf_bytes)
    metadata = _build_metadata(filename, pdf_bytes, markdown)

    if storage.source_exists(metadata["source"]):
        logger.info("Skipping %s — already ingested (source_id=%s)", filename, metadata["source"])
        return {
            "ok":        False,
            "skipped":   True,
            "source_id": metadata["source"],
            "title":     metadata["title"],
        }

    doc = Document(text=markdown, metadata=metadata)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents([doc])

    storage.index_nodes(nodes)

    logger.info("Ingested %d chunks from %s", len(nodes), filename)
    return {
        "ok":        True,
        "skipped":   False,
        "chunks":    len(nodes),
        "source_id": metadata["source"],
        "title":     metadata["title"],
    }


async def ingest_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    storage: VectorRepository,
    parser: PDFParser | None = None,
) -> dict:
    return await asyncio.to_thread(ingest, pdf_bytes, filename, storage, parser)