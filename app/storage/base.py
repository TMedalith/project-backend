from abc import ABC, abstractmethod

from llama_index.core.schema import NodeWithScore


class VectorRepository(ABC):
    @abstractmethod
    def source_exists(self, source_id: str) -> bool:
        pass

    @abstractmethod
    def index_nodes(self, nodes: list) -> None:
        pass

    @abstractmethod
    def retrieve_nodes(self, query: str, top_k: int) -> list[NodeWithScore]:
        pass
