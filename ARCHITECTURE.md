# Architecture: The arXiv Research Agent

This document describes the design of the arXiv Research Agent at production
scale, and how the accompanying proof-of-concept (this repository) implements
a working, scaled-down version of the same design. It answers the three
questions posed in the assessment: ingestion pipeline, parsing strategy,
and handling the ground truth problem.
## Architecture Diagram

<p align="center">
  <img src="assets\architecture_diagram.png"
       alt="Architecture Diagram"
       width="900">
</p>

---

## 1. The Ingestion Pipeline

arXiv cannot be treated as a scrapeable website: it strictly rate-limits its
public API to **3 requests/second**, and blocks clients that behave like
scrapers. The design therefore splits ingestion into two distinct paths with
different data sources, because "load millions of historical papers" and
"stay current day-to-day" are different problems with different constraints.

**Historical cold-start (bulk load).**
arXiv publishes its full corpus - PDFs, LaTeX source, and metadata - as
nightly-updated dumps in a **Requester-Pays S3 bucket** (`arxiv` /
`arxiv-source`), specifically so that large-scale consumers don't have to hit
the rate-limited API millions of times. The historical load pulls directly
from this bucket in batches, with:
- Idempotent, resumable batch jobs (track last-processed shard so a crash
  doesn't require restarting from zero)
- Parallel download workers, since S3 has no per-request rate limit
  comparable to the API's - throughput is bounded by cost and bandwidth, not
  by arXiv's infrastructure
- A queue (e.g. SQS or a lightweight job table) feeding the parsing/embedding
  stage, so ingestion and processing scale independently

**Daily updates (incremental).**
Once caught up, staying current uses the **OAI-PMH harvesting API**
(`export.arxiv.org/oai2`), which is purpose-built for exactly this: querying
"give me everything changed since date X" rather than polling or re-scraping.
A scheduled daily job requests the previous day's delta, respecting the
documented rate limit with a fixed delay between requests (and exponential
backoff with jitter on 429s). This is the same mechanism library aggregators
and academic indexers use, so it's the "polite, sanctioned" path rather than
an adversarial one.

**PoC implementation.** Reproducing S3 bulk-loading and a scheduled OAI-PMH
harvester is out of scope for a prototype - there's no meaningful
architectural difference in loading 10 papers vs. 10,000 via the same client.
Instead, the PoC uses the official **arXiv Atom API**
(`export.arxiv.org/api/query`), the same interface OAI-PMH-style daily deltas
would use in miniature: a single rate-limited client, throttled to the
documented 3 req/s via an explicit delay between calls, fetching metadata and
the LaTeX source (`e-print` endpoint) for the 10 most recent `cs.AI` papers.
The client is written so that swapping "query last 10 papers" for "query
papers since date X" (the daily-delta case) or "iterate an S3 manifest" (the
bulk case) is a matter of changing the input, not the underlying rate-limiting
or retry logic.

---

## 2. The Parsing Strategy

Standard RAG pipelines download the PDF, run it through a generic text
extractor, and chunk on character count. For scientific papers this destroys
exactly the content that matters most: two-column layouts get read
left-to-right across columns (interleaving unrelated paragraphs), and LaTeX
equations rendered to PDF become unrecoverable Unicode soup or vanish
entirely.

**The fix is to not use the PDF as the primary source.** arXiv provides the
original **LaTeX source** for the large majority of submissions via the same
S3 bucket / `e-print` endpoint used for ingestion. LaTeX source is already
structured, machine-readable, and contains equations as verbatim, well-formed
markup (`$...$`, `\begin{equation}...\end{equation}`) rather than rasterized
symbols - there is no OCR or layout-reconstruction problem to solve, because
the structure was never destroyed in the first place.

The parsing strategy follows these steps:
1. **Unpack** the LaTeX source tarball and identify the main `.tex` file
   (handling multi-file projects with `\input`/`\include`).
2. **Chunk on logical structure**, not character count - split at
   `\section` / `\subsection` boundaries, so each chunk corresponds to a real
   unit of meaning (e.g. "Methodology," "Results") rather than an arbitrary
   character window.
3. **Treat equations as atomic, non-splittable units.** An equation
   environment is never split mid-expression; it's kept intact and attached
   to its surrounding sentence(s) for context, so retrieval can return
   "the loss function is defined as `\mathcal{L} = ...`" as one coherent
   chunk rather than fragments.
4. **Tag each chunk with metadata** (arXiv ID, section name) at chunk-creation
   time, so provenance is established before anything touches the vector
   store - traceability is a property of the data model, not something
   reconstructed later at query time.
5. **Fallback for source-unavailable papers.** A minority of submissions omit
   LaTeX source (e.g. scanned or PDF-only submissions). For these, the
   pipeline falls back to layout-aware PDF extraction (`pdfplumber`, which
   preserves column and block structure far better than naive text-stream
   extraction) rather than silently dropping the paper.

**Embeddings.** Chunks are embedded with a local `sentence-transformers`
model (`all-MiniLM-L6-v2`). This is a deliberate architectural choice: math-
and-code-heavy chunk text doesn't require a hosted, general-purpose embedding
API to represent well, and keeping embedding local removes an external
dependency, a cost-per-token, and a rate limit from the hot path of
ingestion - important once ingestion is happening at the scale described in
Section 1.

---

## 3. The "Ground Truth" Problem

arXiv is a pre-print server: nothing is peer-reviewed, and papers routinely
disagree - two papers can both use rigorous methodology and reach opposite
conclusions because they tested different architectures, scales, or
datasets. An agent that retrieves the top-k matching chunks and states the
answer as fact is not summarizing research, it's laundering whichever paper's
phrasing happened to embed closest to the query into unearned authority.

The design treats this as a **retrieval-and-synthesis problem, not a
retrieval-and-recitation problem**:

1. **Retrieve across papers, not just top-k chunks.** The retriever
   (`backed by **FAISS** with a JSON/SQLite metadata
   sidecar carrying `{arxiv_id, title, authors, url, section, chunk_text}`
   per vector) is configured to favor diversity across source papers for a
   given query, not just raw similarity - so a query is answered from
   multiple independent papers when they exist, rather than from five chunks
   of the single best-matching one. Retrieval first ranks chunks by semantic similarity, then re-ranks the candidates to maximize both relevance and diversity across independent papers 

2. **The synthesis prompt is structured to compare, not just answer.**
   The LLM (supporting **Groq** and **OpenAI**) receives the retrieved chunks as
   numbered, attributed sources and is explicitly instructed to:
   - paraphrase and synthesize rather than quote,
   - attach an inline citation marker (`[1]`, `[2]`, ...) to every claim,
     tied to the specific source that supports it, and
   - **surface disagreement explicitly** rather than average it away - e.g.
     "Paper A found GELU converges faster on transformer architectures `[1]`,
     while Paper B reports comparable performance from Swish/SiLU on deeper
     networks `[2]`; the literature does not yet converge on a single
     answer." When retrieved sources conflict, that conflict *is* the
     correct answer, and the prompt is designed to produce it rather than
     resolve it artificially.

3. **Citations are validated, not trusted.** After generation, the response is checked: every `[n]` marker
   must map to a source that was actually in the retrieved context, and the
   answer must contain at least one citation. If validation fails - a
   hallucinated citation number, or an uncited claim - the API returns an
   explicit failure response rather than silently shipping an unsupported
   answer. This makes traceability a system-enforced guarantee, not a
   prompting convention the model can quietly ignore.

4. **No claim is presented as settled fact by default.** The system's
   framing is "here is what the current pre-print literature says, and where
   it agrees or disagrees" - not "here is the answer." This is a closer
   match to what arXiv actually is, and it's the only way to avoid the agent
   hallucinating a false consensus where none exists.

---

## Summary

| Concern | Production design | PoC implementation |
|---|---|---|
| Historical ingestion | S3 bulk load, requester-pays, parallel workers | Out of scope (same client as below, different input) |
| Daily ingestion | OAI-PMH incremental harvest | arXiv Atom API, rate-limited to 3 req/s |
| Parsing | LaTeX source, section + equation-aware chunking, PDF fallback | Implemented as designed |
| Embeddings | Local `sentence-transformers` | Implemented as designed |
| Vector store | FAISS + metadata sidecar | Implemented as designed |
| Synthesis LLM | Pluggable (Groq / OpenAI) | Implemented as designed |
| Contradiction handling | Cross-paper retrieval + comparison-forcing prompt | Implemented as designed |
| Traceability | Per-claim citation, validated post-generation | Implemented as designed |
