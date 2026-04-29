import re

from llama_index.core.schema import NodeWithScore


def extract_cited_indices(text: str) -> set[int]:
    return {int(n) for n in re.findall(r'\[(\d+)\]', text)}


def filter_to_cited(
    nodes: list[NodeWithScore], response_text: str
) -> list[NodeWithScore]:
    cited = extract_cited_indices(response_text)
    filtered = [n for i, n in enumerate(nodes, 1) if i in cited]
    return filtered if filtered else nodes


def extract_sources(nodes: list[NodeWithScore]) -> list[dict]:
    sources = []
    seen: set[str] = set()
    for node in nodes:
        source_id = node.metadata.get("source", "Unknown")
        if source_id in seen:
            continue
        seen.add(source_id)
        sources.append({
            "source":   source_id,
            "title":    node.metadata.get("title", source_id),
            "authors":  node.metadata.get("authors", []),
            "year":     node.metadata.get("year"),
            "doi":      node.metadata.get("doi"),
            "journal":  node.metadata.get("journal"),
            "score":    node.score,
        })
    return sources