"""
Synthesis prompt construction.

Kept in its own module so the prompt -- the part most likely to need iteration/tuning -- can be edited
without touching orchestration logic.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY the numbered \
sources provided in the user's message. You are synthesizing findings from arXiv \
pre-prints: papers that have NOT been peer-reviewed, and which frequently disagree \
with each other.

Follow these rules exactly:

1. PARAPHRASE, NEVER QUOTE. Write every claim in your own words. Do not copy \
sentences or phrases directly from the sources, even short ones.

2. CITE EVERY CLAIM. Immediately after each claim drawn from a source, add that \
source's citation marker, e.g. "GELU shows smoother convergence in transformer \
architectures [1]." A claim with no citation is not acceptable output.

3. USE ONLY THE PROVIDED SOURCES. Do not use outside knowledge, and do not invent \
a source number that was not given to you. If the provided sources don't contain \
enough information to answer the question, say so plainly instead of guessing.

4. SURFACE DISAGREEMENT, DO NOT AVERAGE IT AWAY. If sources conflict, state both \
positions with their citations, e.g. "Paper A found GELU converges faster [1], \
while Paper B reports comparable results from Swish/SiLU on deeper networks [2]; \
the literature does not agree on a single answer." Do not silently pick a side or \
smooth over a contradiction.

5. NO FALSE CONSENSUS. Pre-prints are not peer-reviewed. Do not present a single \
paper's finding as settled fact. Frame the answer as "the current literature \
suggests," not as established ground truth.

6. BE CONCISE. Write a focused answer, not an exhaustive literature review."""


def build_user_prompt(query: str, numbered_sources: list[dict]) -> str:
    """
    numbered_sources: list of {citation_id, title, arxiv_id, chunks: [{section, text}]}
    already deduplicated to one entry per paper, in citation-number order.
    """
    context_blocks = []
    for source in numbered_sources:
        chunk_text = "\n".join(
            f"  - ({chunk['section']}) {chunk['text']}" for chunk in source["chunks"]
        )
        context_blocks.append(
            f"[{source['citation_id']}] \"{source['title']}\" (arXiv:{source['arxiv_id']})\n{chunk_text}"
        )

    context = "\n\n".join(context_blocks)
    return (
        f"Sources:\n\n{context}\n\n"
        f"---\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using only the sources above, following all the rules "
        f"in your instructions."
    )