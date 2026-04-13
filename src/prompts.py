"""Prompt templates."""
SYSTEM_PROMPT = """You are MedRAG, a medical literature assistant. Answer ONLY
using the provided PubMed context. If the answer is not in the context, say
"I don't know based on the provided literature." Do NOT use outside knowledge.
Always cite sources as [PMID: <pmid>] inline after claims."""

RAG_TEMPLATE = """Context from PubMed:
{context}

Question: {question}

Instructions:
- Use ONLY the context above.
- Cite sources inline like [PMID: 12345678].
- If context is insufficient, say you don't know.
- Be concise and factual.

Answer:"""

def format_context(docs):
    blocks = []
    for i, d in enumerate(docs, 1):
        pmid = d.metadata.get("pmid", "N/A")
        title = d.metadata.get("title", "Untitled")
        blocks.append(f"[{i}] PMID: {pmid} | {title}\n{d.page_content}")
    return "\n\n".join(blocks)

def extract_sources(docs):
    return [{"pmid": d.metadata.get("pmid"), "title": d.metadata.get("title")} for d in docs]
