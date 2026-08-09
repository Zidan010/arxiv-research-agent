# arXiv Research Agent

An agent that ingests, parses, and answers questions about recent AI/ML research from arXiv with **paraphrased, per-claim cited answers** and **full source traceability** back to the originating papers.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full design write-up (Part 1 of the assessment).

---

## Table of Contents

- [Overview](#overview)
- [Architecture at a Glance](#architecture-at-a-glance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Option A: Docker (recommended)](#option-a-docker-recommended)
  - [Option B: Local Python](#option-b-local-python)
- [Running the Ingestion Pipeline](#running-the-ingestion-pipeline)
- [Using the API](#using-the-api)
- [Environment Variables](#environment-variables)
- [Known Limitations](#known-limitations)
- [Development Notes](#development-notes)

---

## Overview

Given a natural-language research question, the agent:

1. Retrieves the most relevant chunks of text from a locally indexed set of recent `cs.AI` papers.
2. Synthesizes a **paraphrased** answer (never a verbatim quote) using an LLM, with an inline `[1]`, `[2]`, ... citation marker attached to every claim.
3. Returns the answer alongside a structured `sources` array mapping each citation number back to the exact arXiv ID, title, authors, URL, and text chunks that supported it.
4. When retrieved papers disagree, the answer says so explicitly (e.g. *"Paper A found X [1], while Paper B reports Y [2]; the literature does not agree"*) rather than presenting one paper's finding as settled fact.

Every claim in the answer is verifiable, nothing is asserted without a citation, and every citation is checked against the sources actually retrieved before the response is returned.

---
| # | Criterion | How this project meets it | Code |
|---|---|---|---|
| 1 | **No arXiv scrape** | Uses the official arXiv **Atom API**, throttled to the documented 3 req/s via a shared rate limiter applied before every request. No HTML scraping anywhere. The production-scale design (bulk **S3** cold-start + **OAI-PMH** daily deltas) is detailed in `ARCHITECTURE.md`. | `app/ingestion/arxiv_client.py` |
| 2 | **Respect the math** | Parses the raw **LaTeX source** (not the PDF) for the vast majority of papers. Equations (5 environment types, `$$...$$`, `\[...\]`, `$...$`) are protected as atomic units before chunking, so a chunk boundary can never land mid-expression. A secondary, column-aware **PDF fallback** handles the minority of papers with no LaTeX source available. | `app/parsing/latex_chunker.py`, `app/parsing/pdf_fallback.py` |
| 3 | **Act as a critic, not a parrot** | The synthesis prompt explicitly instructs the LLM to paraphrase (never quote), and to surface disagreement between sources rather than average it away. Retrieval itself favors diversity across papers (a configurable cap on how many chunks may come from one paper) rather than just raw top-k similarity, so contradictory evidence has a chance to surface in the first place. | `app/rag/prompts.py`, `app/rag/synthesizer.py`, `app/vectorstore/faiss_store.py` |
| 4 | **Provide traceability** | The API returns structured JSON: every claim's citation number maps to an explicit `sources[]` entry containing arXiv ID, title, authors, URL, and the exact chunk text used. Citations are **validated after generation** — an answer with zero citations, or a citation number that wasn't actually among the retrieved sources, is rejected outright rather than returned. | `app/rag/synthesizer.py`, `app/api/schemas.py` |
| 5 | **Production readiness** | Fully containerized: `docker-compose up` starts the API; the ingestion pipeline runs as a separate, on-demand profile-gated service. No manual model downloads (the embedding model downloads on first use and is cached in the image/volume). Environment variables are validated at startup via `pydantic-settings`, with graceful degradation (clear 503, not a crash) if the index or an LLM API key isn't ready yet. | `Dockerfile`, `docker-compose.yml`, `app/config.py` |

---

## Architecture at a Glance

```
 ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌──────────────┐
 │  Ingest   │──▶│   Parse   │──▶│  Embed    │──▶│ Build Index  │
 │ (arXiv    │   │ (LaTeX /  │   │ (local    │   │ (FAISS +     │
 │  Atom API)│   │  PDF      │   │  sentence-│   │  metadata    │
 │           │   │  fallback)│   │  transf.) │   │  sidecar)    │
 └───────────┘   └───────────┘   └───────────┘   └──────────────┘
                                                          │
                                                          ▼
                                            ┌───────────────────────┐
   User ── POST /api/research/query ──────▶│   RAG Synthesizer      │
                                            │  retrieve → group by   │
                                            │  paper → prompt (Groq) │
                                            │     → validate         │
                                            │  citations             │
                                            └───────────────────────┘
                                                          │
                                                          ▼
                                       { answer, sources[], status }
```

Full rationale for every one of these design decisions including the production-scale ingestion design (S3 bulk load + OAI-PMH daily deltas), the equation-atomicity approach, and the contradiction-handling strategy is in **[`ARCHITECTURE.md`](./ARCHITECTURE.md)**.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| API framework | **FastAPI** | Async, automatic OpenAPI docs, Pydantic-native request/response validation |
| Ingestion | Official **arXiv Atom API**, rate-limited client | The only sanctioned, non-scraping path; matches the "hire signal" in the assessment |
| Parsing | Custom **LaTeX-aware chunker** + column-aware PDF fallback | Preserves equations and section structure instead of destroying them |
| Embeddings | Local **sentence-transformers** (`all-MiniLM-L6-v2`) | No API key, no rate limit, no cost — fully decoupled from the LLM provider |
| Vector store | **FAISS** (`IndexFlatIP`) + JSON metadata sidecar | Embedded, file-based, zero extra infrastructure; explicitly acceptable per the assessment |
| Synthesis LLM | **Groq** implemented, selected via `LLM_PROVIDER` | Groq as the free-tier default |
| Containerization | Docker + Docker Compose | Single-command startup, no manual setup |
| Config | `pydantic-settings` | Typed, validated, defaulted configuration from environment variables |

---

## Project Structure

```
arxiv-research-agent/
├── ARCHITECTURE.md            # full design rationale
├── README.md                  
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── data/                      # generated at runtime (gitignored): raw/, processed/, vector_store/
├── scripts/
│   ├── ingest.py               # fetch papers from arXiv
│   ├── parse.py                 # LaTeX/PDF -> chunks
│   ├── embed.py                  # chunks -> vectors
│   └── build_index.py             # vectors + metadata -> FAISS index
├── app/
│   ├── main.py                 # FastAPI entrypoint, startup wiring
│   ├── config.py                # typed settings (env-driven)
│   ├── logging_config.py
│   ├── api/                     # request/response schemas + the query endpoint
│   ├── ingestion/                # rate-limited arXiv client
│   ├── parsing/                   # LaTeX chunker + PDF fallback
│   ├── embeddings/                 # local embedding wrapper
│   ├── vectorstore/                 # FAISS store (interface + implementation)
│   ├── llm/                          # Groq providers (interface + factory)
│   └── rag/                           # retrieval -> prompting -> citation validation
└── tests/                       # (planned — see Development Notes)
```

---

## Getting Started

### Prerequisites

- A **Groq API key** (free tier): https://console.groq.com 
- Docker + Docker Compose (recommended path), **or** Python 3.12 (local path)

### Option A: Docker (recommended)

```bash
# 1. Configure
cp .env.example .env
# edit .env and set GROQ_API_KEY=your_key_here

# 2. Start the API
docker-compose up
# API is now live at http://localhost:8000 — GET /health should return {"status": "ok"}
# (the index hasn't been built yet, so /api/research/query will return a 503 until step 3)

# 3. Run the ingestion pipeline (separate, on-demand — see note below)
docker-compose --profile pipeline run --rm pipeline
```

The vector store is **FAISS** — an embedded, file-based index rather than a separate database server, so there's no second "vector-db" container to start. The `./data` folder is bind-mounted, so everything the pipeline produces (raw papers, parsed chunks, embeddings, the FAISS index) persists on your host and survives container restarts.

The `pipeline` service is intentionally **not** started by plain `docker-compose up` ingestion is a one-time/periodic job, not something that should re-run every time the API container restarts.

### Option B: Local Python

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GROQ_API_KEY=your_key_here

python -m scripts.ingest
python -m scripts.parse
python -m scripts.embed
python -m scripts.build_index

uvicorn app.main:app --reload
```

---

## Running the Ingestion Pipeline

The four stages run independently and in order, each reading the previous stage's output:

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1. Ingest | `python -m scripts.ingest` | arXiv Atom API | `data/raw/papers.json`, `data/raw/sources/*.tar.gz` |
| 2. Parse | `python -m scripts.parse` | `data/raw/papers.json` | `data/processed/chunks.json` |
| 3. Embed | `python -m scripts.embed` | `data/processed/chunks.json` | `data/processed/embeddings.npy`, `embeddings_index.json` |
| 4. Build index | `python -m scripts.build_index` | all of the above | `data/vector_store/index.faiss`, `metadata.json` |

By default this fetches the **10 most recent `cs.AI` papers** (configurable via `ARXIV_CATEGORY` / `ARXIV_MAX_RESULTS`).

---

## Using the API

### `GET /health`

Basic liveness check — returns `200` as soon as the container is up, independent of whether the index has been built yet.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### `POST /api/research/query`

```bash
curl -X 'POST' \
  'http://localhost:8000/api/research/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "are Large language models now serve as reasoning engines?"
}'
```

**Example response:**

```json
{
  "query": "are Large language models now serve as reasoning engines?",
  "status": "ok",
  "answer": "The current literature suggests that large language models are being used as reasoning engines, as they can answer scientific questions and justify their decisions through long chains of thought [1]. This is evident in their ability to solve problems and provide explanations for their answers, although the accuracy of these explanations can be fragile and dependent on the context in which the question is asked [1]. However, the evaluation of these models' reasoning capabilities is still a challenging task, particularly in the context of explainable AI (XAI), where the concept of explanations is ill-defined and highly sensitive to design choices [2]. Nevertheless, research in this area is ongoing, with some studies focusing on the development of systems that can construct analytic semantic schemas from relational data, which can help improve the accuracy and reliability of large language models [3].",
  "sources": [
    {
      "citation_id": 1,
      "arxiv_id": "2608.06377",
      "title": "Learning When to Trust via Selective Context Preference Optimization",
      "authors": [
        "Xian Sun",
        "Wei Chow",
        "Yingshuo Wang",
        "Junhao Liu",
        "Wei Gao",
        "Qing Wu",
        "Lingdong Kong"
      ],
      "url": "http://arxiv.org/abs/2608.06377v1",
      "chunks_used": [
        {
          "section": "Introduction",
          "text": "Large language models now serve as reasoning engines: they answer scientific questions and justify their decisions through long chains of thought \\citep{wei2022chain,kojima2022large}, and broad benchmarks have made this progress measurable across knowledge, reasoning, and holistic metric suites \\citep{hendrycks2021measuring,srivastava2022beyond,suzgun2022challenging,liang2022holistic,wang2024mmlupro}. But those benchmarks score a model under a single context, and a final answer is fragile to what surrounds the question. A model that solves a problem cleanly will often abandon its correct answer the moment the prompt carries a plausible but wrong suggestion. Such cues steer predictions without being faithfully reflected in the model's stated reasoning \\citep{turpin2023language,chen2025reasoning}, and preference-based alignment can make matters worse by rewarding agreement with the user over truth \\citep{sharma2023towards}. Fig.~\\ref{fig:teaser} shows the effect across models: an official-looking but wrong hint flips even frontiers from right to wrong.\n\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/fig1.pdf}\n\\vspace{-0.59cm}"
        },
        {
          "section": "Existing-Benchmark Cases",
          "text": "Cases 7 and 8 are drawn from existing benchmark sources rather than the human-authored subset. The model rejects a domain-knowledge answer key that states a confident but incorrect fact (Case 7) and overrides an answer key on a multi-step word problem, recomputing the result from the problem statement (Case 8). Because these items originate from public benchmarks, they show the behavior is not an artifact of our authoring style or of a particular prompt template."
        }
      ]
    },
    {
      "citation_id": 2,
      "arxiv_id": "2608.06351",
      "title": "Challenges in Evaluating Explanation Methods for Static and Evolving Data",
      "authors": [
        "Jerzy Stefanowski"
      ],
      "url": "http://arxiv.org/abs/2608.06351v1",
      "chunks_used": [
        {
          "section": "Difficulties in Evaluating XAI Methods",
          "text": "\\label{sec2:methods}\n\nEvaluating the explanations is challenging, given the wide range of numerous proposals and ill-defined tasks of XAI \\cite{moshkovitz2026explainability}.\n\nFirstly,  the XAI literature has produced a wide range of mainly post-hoc techniques to approximate different aspects of model performance with many methodological paradigms. However, they differ very much in terms of the types of data (tables, images, text, time series, etc.), the representations of explanations provided, its focus on different audiences, and other aspects.  As discussed in the paper \\cite{moshkovitz2026explainability}, despite substantial methodological diversity, XAI approaches are not easy to apply, are highly sensitive to arbitrary design choices, and are often misused in practice. In general, they are  used correctly by model developers or domain experts often than by other users."
        },
        {
          "section": "Difficulties in Evaluating XAI Methods",
          "text": "The concept of XAI is rather ill-defined; see the overview of various definitions in \\cite{arrieta2020explainable}. Here, we follow the definition  \\cite{guidotti2018survey}: \\textit{Explainable-AI explores and investigates methods to produce or complement AI models to make accessible and interpretable the internal logic and the outcome of the algorithms, making such process understandable by humans}. \nSo, we can distinguish between \\textit{global explanations} and \\textit{local explanations} --  which we will further understand as attempts to identify the reasons behind a black-box ML modelâ€™s decision for a specific instance. Further on, we will consider local explanations.\n\nAnother difficult aspect is that explanations are intended for people and should be useful to them. This is also  ambiguous, as it depends strongly on the type of recipient, including their domain knowledge, preferences, and the specific task being solved. Following \\cite{byrne2019counterfactuals}, currently proposed XAI methods take too much into account the perspective of the AI system developers, not other users."
        }
      ]
    },
    {
      "citation_id": 3,
      "arxiv_id": "2608.06331",
      "title": "Tytan: Interactive Neurosymbolic Construction of Analytic Semantic Schemas from Relational Data",
      "authors": [
        "Donna Hooshmand",
        "Shubham Shahi",
        "Cameron Barrie",
        "Abhratanu Dutta",
        "Marko Sterbentz",
        "Harper Pack",
        "Kristian J. Hammond"
      ],
      "url": "http://arxiv.org/abs/2608.06331v1",
      "chunks_used": [
        {
          "section": "Preamble",
          "text": "\\maketitle\n\\begin{abstract}\n\nFrom natural-language query interfaces to automated report generation, data analysis tools\nneed a description of the data: the real-world entities it contains, which columns function as measures or identifiers, and how tables connect into units of analysis. Today, this semantic layer is usually written by hand. This is a knowledge-acquisition bottleneck that limits the scalability of analytic systems, keeps non-technical users dependent on experts, and is itself error-prone.\nWe present \\textsc{Tytan}, a system for automatically constructing an \\emph{analytic semantic schema} from a relational database and, when available, a short user-provided description. \\textsc{Tytan} combines symbolic analysis of the database with LLM-based semantic inference for entity proposal, role assignment, and naming. When the evidence leaves a decision ambiguous, \\textsc{Tytan} asks the user a targeted natural-language question."
        },
        {
          "section": "Introduction",
          "text": "Organizations often place a semantic layer between raw databases and the people or systems that use them. Instead of describing data only through tables, columns, and foreign keys, this layer represents the data in terms of real-world entities, attributes, and relationships \\citep{chen1976entity,chen2012business}.\n\nThis intermediate layer is what allows a business-intelligence tool to interpret a request such as \"average fire size per state\" or a natural language interface to translate \"which professors taught the most classes?\" into appropriate joins and aggregations. Without this context, query systems would need to infer meaning directly from the schema, which is often ambiguous. As a result, LLM-based agents can misinterpret tables and relationships, and production natural-language-to-SQL systems remain unreliable on real-world databases that do not provide enough semantic information to resolve user intent \\citep{floratou2024nl2sql, kim2020natural, li2024dawn}."
        }
      ]
    }
  ],
  "error": null
}
```

**Possible `status` values:**

| Status | Meaning |
|---|---|
| `ok` | Answer generated and successfully cited |
| `no_results` | Nothing relevant found in the index for this query |
| `validation_failed` | The LLM's answer had zero citations, or cited a source that wasn't actually retrieved — withheld rather than returned unverified |
| `llm_error` | The configured LLM provider failed after retries |

A `503` at the HTTP level (rather than one of the statuses above) means the service itself isn't ready yet most commonly because the ingestion pipeline hasn't been run. The response body explains exactly what to do.

---

## Environment Variables

All configuration is documented in [`.env.example`](./.env.example) with inline comments. The essentials:

| Variable | Default | Notes |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq`|
| `GROQ_API_KEY` | *(empty)* | Required Free at console.groq.com |
| `ARXIV_CATEGORY` | `cs.AI` | arXiv category to ingest |
| `ARXIV_MAX_RESULTS` | `10` | Number of papers per ingestion run |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model, no API key needed |
| `RETRIEVAL_TOP_K` | `6` | Chunks retrieved per query |
| `RETRIEVAL_MAX_CHUNKS_PER_PAPER` | `2` | Diversity cap — favors multiple papers over one dominant match |

---

## Known Limitations

Documented here in the interest of the same honesty maintained throughout development:

- **LaTeX macro expansion**: custom `\newcommand` macros defined in a paper's preamble are not expanded in chunk text (only the `\begin{document}...\end{document}` body is parsed, to avoid chunking macro-definition noise). The LaTeX remains valid and generally interpretable by the LLM from context, but isn't rendered to a human-readable symbol.
- **PDF fallback column detection**: relies on a horizontal gap between columns to separate them. Real published two-column papers reliably have a visible gutter, but a sufficiently narrow one could, in principle, cause the last word of one column and the first word of the next to merge — a known limitation of gap-based word detection, documented inline in `app/parsing/pdf_fallback.py`.
- **No conversation memory**: each query is independent; there's no multi-turn follow-up context (out of scope for this assessment).

---

