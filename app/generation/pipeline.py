import asyncio
from dataclasses import dataclass

from llama_index.core.schema import NodeWithScore

from app.retrieval.retrieval import retrieve_nodes
from app.generation.context import format_context, build_messages
from app.storage.base import VectorRepository


@dataclass
class PipelineResult:
    nodes: list[NodeWithScore]
    messages: list


class RAGPipeline:
    def __init__(self, storage: VectorRepository) -> None:
        self._storage = storage

    async def prepare(self, question: str, top_k: int) -> PipelineResult:
        nodes = await asyncio.to_thread(
            retrieve_nodes, self._storage, question, top_k
        )
        context = format_context(nodes)
        messages = build_messages(context, question)
        return PipelineResult(nodes=nodes, messages=messages)
