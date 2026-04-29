from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.vector_stores.postgres import PGVectorStore
import sqlalchemy

from app.config import settings
from app.storage.base import VectorRepository


class PGStorage(VectorRepository):

    def __init__(self) -> None:
        self._vector_store = PGVectorStore.from_params(
            host=settings.pg_host,
            port=settings.pg_port,
            user=settings.pg_user,
            password=settings.pg_password,
            database=settings.pg_database,
            table_name=settings.pg_table,
            embed_dim=settings.embed_dimensions,
        )
        storage_context = StorageContext.from_defaults(vector_store=self._vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            storage_context=storage_context,
        )
        self._storage_context = storage_context

    def source_exists(self, source_id: str) -> bool:
        engine = self._vector_store._engine
        table_name = settings.pg_table
        with engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(
                    f"SELECT 1 FROM data_{table_name} WHERE metadata_->>'source' = :src LIMIT 1"
                ),
                {"src": source_id},
            )
            return result.fetchone() is not None

    def index_nodes(self, nodes: list) -> None:
        self._index.insert_nodes(nodes)

    def retrieve_nodes(self, query: str, top_k: int = 20) -> list[NodeWithScore]:
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=top_k,
        )
        return retriever.retrieve(QueryBundle(query))
