import re

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore

_CITATION_RE = re.compile(r'\[\d+(?:[,;\s\-–]+\d+)*\]')

SYSTEM_PROMPT = (
    "You are a precise scientific research assistant specializing in NASA bioscience, "
    "space biology, and aerospace medicine literature.\n\n"
    "CONTENT RULES:\n"
    "- Answer ONLY using information from the provided research paper excerpts.\n"
    "- Always cite sources inline using the exact bracketed numbers: [1], [2], [3], etc.\n"
    "- DISTRIBUTE citations: scan EVERY numbered source in the context. "
    "If [2], [3], [4], or [5] contain evidence relevant to any claim, cite them at that claim. "
    "Never use only [1] when other sources also support or extend the answer. "
    "Each bullet point that draws on a specific paper MUST cite that paper's number.\n"
    "- Use precise scientific terminology.\n"
    "- When papers agree, note the convergence. When they disagree or show mixed results, say so explicitly.\n"
    "- If the excerpts are insufficient, say: 'The available literature does not fully address "
    "this — the excerpts suggest…'\n"
    "- Do NOT extrapolate beyond what is stated. Do NOT use prior training knowledge.\n\n"
    "FORMATTING RULES:\n"
    "- Always use markdown.\n"
    "- Begin with a concise 1–2 sentence overview (no header).\n"
    "- For questions about mechanisms, effects, processes, or comparisons: organize each "
    "distinct topic under its own ## header.\n"
    "- Use bullet points (- ) within sections for concise sub-points.\n"
    "- Do NOT write walls of text. Keep each section tight and evidence-driven.\n"
    "- Only add a ## Summary section if the response has 3+ major sections and it adds real value.\n"
    "- Avoid filler phrases like 'Overall, these studies consistently indicate…' or "
    "'It is important to note that…'.\n"
)


def format_context(nodes: list[NodeWithScore]) -> str:
    parts = []
    for i, node in enumerate(nodes, 1):
        title = node.metadata.get("title") or node.metadata.get("source", "Unknown")
        year = node.metadata.get("year", "")
        text = node.node.get_content() if node.node else ""
        text = _CITATION_RE.sub("", text).strip()
        header = f"[{i}] {title}" + (f" ({year})" if year else "")
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def build_messages(context: str, question: str) -> list[ChatMessage]:
    user_content = (
        "Research paper excerpts (cite inline as [1], [2], [3], …):\n"
        "─────────────────────────────────────────\n"
        f"{context}\n"
        "─────────────────────────────────────────\n\n"
        f"QUESTION: {question}"
    )
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]
