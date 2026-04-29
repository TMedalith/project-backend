import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from app.config import settings
from app.routes.query import router as query_router
from app.routes.ingest import router as ingest_router
from app.storage.vector_db import PGStorage
from app.generation.pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    storage = PGStorage()
    app.state.storage = storage
    app.state.pipeline = RAGPipeline(storage)
    yield


app = FastAPI(title="Raccly API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://raccly.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(query_router)
app.include_router(ingest_router)