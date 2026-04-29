import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from llama_index.core import Settings

from app.models import QueryRequest
from app.generation.sources import filter_to_cited, extract_sources

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/query-stream")
async def query_stream(request: Request, body: QueryRequest) -> StreamingResponse:
    try:
        pipeline = request.app.state.pipeline
        result = await pipeline.prepare(body.question, body.top_k)

        async def event_generator():
            full_response = ""
            try:
                async for chunk in await Settings.llm.astream_chat(result.messages):
                    token = chunk.delta or ""
                    if token:
                        full_response += token
                        yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            except Exception:
                logger.exception("LLM streaming error")
                yield f"data: {json.dumps({'type': 'error', 'content': 'Stream interrupted'})}\n\n"
                return

            cited_nodes = filter_to_cited(result.nodes, full_response)
            sources = extract_sources(cited_nodes)
            yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception:
        logger.exception("Query pipeline error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health() -> dict:
    return {"status": "healthy"}