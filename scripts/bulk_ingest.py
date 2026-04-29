import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _setup_llama_index():
    from llama_index.core import Settings
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from app.config import settings

    Settings.embed_model = OpenAIEmbedding(
        model=settings.embed_model,
        dimensions=settings.embed_dimensions,
        api_key=settings.openai_api_key,
    )
    Settings.llm = OpenAI(
        model=settings.main_llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.openai_api_key,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bulk ingest PDFs into Raccly.")
    parser.add_argument("--dir", required=True, help="Directory with PDF files")
    parser.add_argument("--log", default="ingest_log.json", help="Output log path")
    parser.add_argument("--parser", default=None, help="PDF parser (default: from config)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    pdf_dir = Path(args.dir)
    if not pdf_dir.is_dir():
        logger.error("'%s' is not a valid directory.", pdf_dir)
        sys.exit(1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in '%s'.", pdf_dir)
        sys.exit(1)

    logger.info("Found %d PDF files.", len(pdfs))

    _setup_llama_index()

    from app.storage.vector_db import PGStorage
    from app.ingest.pipeline import ingest
    from app.ingest.parsers import get_parser

    storage = PGStorage()
    parser = get_parser(args.parser)

    results = []
    ok_count = skipped_count = error_count = 0

    for i, pdf_path in enumerate(pdfs, 1):
        prefix = f"[{i}/{len(pdfs)}] {pdf_path.name}"
        logger.info("%s — starting", prefix)
        t0 = time.perf_counter()

        try:
            pdf_bytes = pdf_path.read_bytes()
            result = ingest(pdf_bytes, pdf_path.name, storage, parser)
            elapsed = time.perf_counter() - t0

            if result.get("skipped"):
                skipped_count += 1
                logger.info("%s — skipped (already ingested) %.1fs", prefix, elapsed)
            else:
                ok_count += 1
                logger.info(
                    "%s — ok  chunks=%d  %.1fs",
                    prefix, result["chunks"], elapsed,
                )

            results.append({"file": pdf_path.name, "elapsed_s": round(elapsed, 1), **result})

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            error_count += 1
            logger.error("%s — FAILED (%.1fs): %s", prefix, elapsed, exc, exc_info=True)
            results.append({
                "file":      pdf_path.name,
                "ok":        False,
                "skipped":   False,
                "error":     str(exc),
                "elapsed_s": round(elapsed, 1),
            })

    log_path = Path(args.log)
    log_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    logger.info(
        "Done. ok=%d  skipped=%d  errors=%d  log=%s",
        ok_count, skipped_count, error_count, log_path,
    )
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
