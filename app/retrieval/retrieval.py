import logging
from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore, QueryBundle

from app.config import settings
from app.storage.base import VectorRepository

logger = logging.getLogger(__name__)


class NodePostProcessor(ABC):
    @abstractmethod
    def process(self, nodes: list[NodeWithScore], query: str) -> list[NodeWithScore]:
        ...


class SimilarityFilter(NodePostProcessor):
    def __init__(self, cutoff: float) -> None:
        self._cutoff = cutoff

    def process(self, nodes: list[NodeWithScore], query: str) -> list[NodeWithScore]:
        return [n for n in nodes if (n.score or 0.0) >= self._cutoff]


class CohereReranker(NodePostProcessor):
    def __init__(self, api_key: str, model: str, top_n: int) -> None:
        from llama_index.postprocessor.cohere_rerank import CohereRerank
        self._reranker = CohereRerank(api_key=api_key, model=model, top_n=top_n)

    def process(self, nodes: list[NodeWithScore], query: str) -> list[NodeWithScore]:
        try:
            return self._reranker.postprocess_nodes(
                nodes, query_bundle=QueryBundle(query)
            )
        except Exception:
            logger.warning("Reranking failed, falling back to vector scores", exc_info=True)
            return nodes


class TopKTrimmer(NodePostProcessor):
    def __init__(self, top_k: int) -> None:
        self._top_k = top_k

    def process(self, nodes: list[NodeWithScore], query: str) -> list[NodeWithScore]:
        return nodes[:self._top_k]


class _PostProcessingChain:
    def __init__(self, processors: list[NodePostProcessor]) -> None:
        self._processors = processors

    def run(self, nodes: list[NodeWithScore], query: str) -> list[NodeWithScore]:
        for processor in self._processors:
            nodes = processor.process(nodes, query)
        return nodes


def _build_chain(top_k: int) -> _PostProcessingChain:
    processors: list[NodePostProcessor] = [SimilarityFilter(settings.similarity_cutoff)]
    if settings.cohere_api_key:
        processors.append(CohereReranker(
            settings.cohere_api_key,
            settings.rerank_model,
            settings.rerank_top_n,
        ))
    processors.append(TopKTrimmer(top_k))
    return _PostProcessingChain(processors)


def retrieve_nodes(
    storage: VectorRepository,
    question: str,
    top_k: int,
) -> list[NodeWithScore]:
    candidates = storage.retrieve_nodes(question, top_k=top_k * 2)
    return _build_chain(top_k).run(candidates, question)
