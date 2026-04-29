import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from app.ingest.pipeline import ingest_pdf_bytes

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ingest")
async def ingest_pdf(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos PDF")

    try:
        pdf_bytes = await file.read()
        storage = request.app.state.storage
        result = await ingest_pdf_bytes(pdf_bytes, file.filename, storage)
        return result
    except Exception:
        logger.exception("Ingest error")
        raise HTTPException(status_code=500, detail="Error procesando el PDF")