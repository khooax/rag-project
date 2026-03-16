# Singapore Employment Rights Advisory Chatbot

A RAG chatbot that answers questions about Singapore employment and workplace rights

The goal of this project is a chatbot that:

1. Answers questions using only official sources (Ministry of Manpower, CPF Board, and Tripartite Advisory), citing the specific document/page every claim comes from
3. Correctly refuses to answer off-topic questions rather than hallucinating
4. Handles informal and colloquial queries, including Singlish and abbreviations

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Design choices](#design-decisions)
  - [Chunking Strategy](#chunking-strategy)
  - [Retrieval: Top-K Selection](#retrieval-top-k-selection)
  - [Query Understanding Pipeline](#query-understanding-pipeline)
  - [Out-of-Scope Detection: Tiered Router](#out-of-scope-detection-tiered-router)
- [Data Pipeline](#data-pipeline)
  - [Web Crawler Design](#web-crawler-design)
  - [Chunk and Embed](#chunk-and-embed)
- [Evaluation Framework](#evaluation-framework)
  - [Metric Definitions](#metric-definitions)
  - [LLM-as-Judge Design and Limitations](#llm-as-judge-design-and-limitations)
  - [Evaluation Results](#evaluation-results)
- [Ablation Studies](#ablation-studies)
  - [1. RAG vs No-RAG Baseline](#1-rag-vs-no-rag-baseline)
  - [2. Chunk Size](#2-chunk-size)
  - [3. Top-K Retrieved Chunks](#3-top-k-retrieved-chunks)
  - [4. Query Rewriting Robustness](#4-query-rewriting-robustness)
  - [5. Out-of-Scope Detection Methods](#5-out-of-scope-detection-methods)
- [Project Structure](#project-structure)
- [Setup and Running](#setup-and-running)
- [Limitations](#limitations)

---

## Architecture Overview

```
User query
    |
    v
Query Understanding (query_understanding.py)
    |-- Tier 1: Semantic similarity against labelled examples (~30ms)
    |-- Tier 2: LLM judge for uncertain cases (~500ms, called rarely)
    |-- Query rewriting: typo fix, Singlish normalisation, abbreviation expansion
    v
Out-of-scope? --> Return redirect response (no LLM call)
    |
    v
Retrieval (ChromaDB + all-MiniLM-L6-v2)
    |-- Embed cleaned query
    |-- Cosine similarity search, top-k=5 chunks
    v
Generation (Llama 3.1 8B via Groq, temperature=0)
    |-- Strict system prompt: only use retrieved context
    |-- Mandatory inline citations [Source: ...]
    |-- Explicit fallback: "I don't have enough information"
    v
Response + source documents + query trace
```

---

## Tech stack
| Component | Choice | Why | 
|---|---|---|
| LLM | Llama 3.3 70B via Groq | Free API, low latency inference | 
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) | Lightweight, runs locally, no API cost
| Vector DB | ChromaDB | Serverless, local 
| Framework | LandChain | RetrievalQA + prompt templating
| Frontend | Streamlit | Rapid UI prototyping
| Query rewriter | Llama 3.1 8B via Groq | Cheaper model sufficient for rewriting

Notes: 
* **LLM**: Temperature=0 - Deterministic decoding for all generation calls, since employment law questions have correct answers and there is no value in stochastic sampling.
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` produces 384-dimensional embeddings and runs on CPU in approximately 10ms per query after the initial model load. For structured legal text with specific terminology ("notice period", "CPF ordinary wage ceiling"), smaller models perform competitively with larger ones because the domain vocabulary is highly distinctive. Embeddings do not need to capture subtle semantic nuance — they need to cluster "notice period" queries near "notice period" chunks, which MiniLM does reliably.
* **Vector DB**: ChromaDB persists to a local directory (`./db`). The entire vector index is committed to the repo, so the HuggingFace Spaces deployment does not require a database server or network call for retrieval. In comparison with managed vector databases (Pinecone, Weaviate), which are appropriate for production systems requiring multi-writer updates/horizontal scaling/real-time indexing, ChromaDB's limitation is single-process access and no built-in hybrid search. For a read-only demonstration with fewer than 1000 chunks, these limitations are irrelevant. 

---
## Design choices

### Chunking Strategy

Documents are split using `RecursiveCharacterTextSplitter` with chunk size 500 tokens and overlap 100 tokens.

**Why 500 tokens.** Employment Act provisions are structured as numbered rules with sub-clauses. At 256 tokens, these components split across chunks, leaving each chunk without sufficient context. At 1024 tokens, a single chunk may cover multiple unrelated provisions, reducing retrieval precision because the embedding represents a mixture of topics.

**Why 100 tokens overlap.** Without overlap, a key sentence that falls exactly at a chunk boundary appears in neither adjacent chunk. An overlap of 100 tokens (approx 2-3 sentences) ensures that boundary content appears in both chunks, at the cost of modest index size increase.

**Why RecursiveCharacterTextSplitter.** It splits on paragraph boundaries first, then sentence boundaries, then word boundaries. This respects the document structure of MOM web pages, which use `\n\n` to separate provisions. Hard character splitting would cut mid-sentence.

The specific values (500, 100) were validated in the ablation study in section [Chunk Size](#2-chunk-size).

---

### Retrieval: Top-K Selection

The pipeline retrieves k=8 chunks per query.

Small k is insufficient for questions that span multiple provisions. A question like "what happens if I am retrenched" requires chunks covering notice period, retrenchment benefit, and CPF withdrawals - three topics that are unlikely to occur in a single chunk.

Large k introduces noise. When irrelevant chunks enter the context window, the LLM must distinguish relevant from irrelevant content. For an 8B model this is unreliable, and the additional tokens increase latency.

k=8 was validated in the ablation study in section [Top-K Retrieved Chunks](#3-top-k-retrieved-chunks).

---

### Query Understanding Pipeline

`query_understanding.py` runs as a preprocessing stage before retrieval. It handles three classes of imperfect queries common in Singapore:

**Stage 1 — Abbreviation expansion and Singlish normalisation (rule-based, instant)**

Abbreviations: OT (overtime), MC (medical certificate/sick leave), WP (Work Permit) etc. These are expanded to their full forms before embedding as embedding models handle them poorly out of the box.

Singlish and informal constructions: "boss never pay me salary", "kena retrench". These are normalised to standard English before embedding. 

The rule set covers common patterns; residual informal language is handled by Stage 3.

**Stage 3 — LLM rewriting (conditional, ~500ms)**

Only triggered when Stages 1 and 2 leave detectable informality (heuristics: abbreviations, short vague queries, known Singlish markers, typos). The LLM rewrites the cleaned query into formal English suitable for legal document retrieval. Results are `lru_cache`d so repeated queries after rewriting are free.

The ablation in section [Query Rewriting Robustness](#4-query-rewriting-robustness) quantifies the retrieval precision improvement for each category of imperfect query.

---

### Out-of-Scope Detection: Tiered Router

`scope_router.py` implements a two-tier classifier to reject answering out of scope queries

**Tier 1 — Semantic nearest-neighbour.** The query is embedded using the same MiniLM model already loaded for retrieval. Cosine similarity is computed against 25 labelled in-scope examples and 20 labelled out-of-scope examples, both embedded at startup. The decision uses a margin threshold: if `max_in_sim - max_out_sim > 0.05`, the query is classified confidently. Nearest-neighbour is used rather than centroid as the out-of-scope example set is deliberately diverse (recipes, coding, sports, finance), and the centroid of diverse examples is a poor representative.

**Tier 2 — LLM judge.** Used only when Tier 2 cannot produce a confident margin. The LLM judge uses a structured prompt with explicit in-scope and out-of-scope categories and returns JSON. If the judge fails, the query is passed through (over-blocking is treated as the more costly error for a citizen-facing system). 

The ablation in section [Out-of-Scope Detection Methods](#5-out-of-scope-detection-methods) benchmarks all three approaches against a 40-query test set with labelled ground truth.

---

## Data Pipeline

We use the Employment Act PDF as well as CPF and MOM websites (scraped with a web crawler) for our dataset. 

### Web Crawler Design

`ingest.py` implements a BFS crawler starting from three MOM seed URLs:

```
https://www.mom.gov.sg/employment-practices
https://www.mom.gov.sg/workplace-safety-and-health
https://www.mom.gov.sg/passes-and-permits
```

MOM's website uses a three-level hierarchy:

```
Level 1: /employment-practices                           (section landing)
Level 2: /employment-practices/employment-act            (subsection landing)
Level 3: /employment-practices/employment-act/who-is-covered  (article)
```

Since Levels 1 and 2 are navigation pages containing only a `media-grid` of links to sub-pages, a naive crawler that attempts to extract prose from every page will produce zero usable documents from all navigation pages, and will not follow their links to articles at Level 3. 

The solution is page classification before content extraction. The crawler checks the tag on MOM's ASP NET rendering: 

- Landing pages contain a meta tag `<meta name="data_template" content="section-landing-template"/>`. Follow all outbound links, discard the page itself
- Article pages: extract prose content from `#MainContent`, follow any inline links

This allows recursive BFS to depth 3+ without requiring manual URL curation.

Additional pages from cpf.gov.sg are fetched directly.

### Chunk and Embed

After crawling, documents are split and embedded in a single pass:

1. `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)` splits each document
2. `all-MiniLM-L6-v2` embeds all chunks
3. ChromaDB stores vectors and metadata (source URL, document title) in `./db`

Source metadata is preserved through to the final answer, enabling inline citations like `[Source: Employment Act - Part IV Working Hours]`.

---

## Evaluation Framework

`eval.py` measures ten metrics across a 20-question test set with human-verified ground truth answers.

### Metric Definitions

**Semantic Similarity**

Cosine similarity between the embedding of the generated answer and the ground truth answer, using the same MiniLM model as the retriever. Range 0 (semantically very different) to 1 (semantically identical), 0.7 (related but differently worded).

**LLM Judge — Faithfulness**

A structured prompt asks the judge LLM to score 0.0-1.0 whether every factual claim in the generated answer is supported by or inferable from the retrieved context chunks. A score below 0.7 indicates the answer went beyond its sources. The judge returns JSON: `{"score": float, "reason": str}`.

Faithfulness measures whether the LLM stayed within its context. It does not measure whether the answer is correct — a highly faithful answer to the wrong context would score 1.0.

**LLM Judge — Correctness**

A structured prompt asks the judge to score 0.0-1.0 whether the generated answer correctly addresses the question relative to the ground truth. This is the primary quality signal. 1.0 = fully correct, 0.7 = correct with minor omissions, 0.5 = partially correct, 0.3 = right topic wrong facts, 0.0 = incorrect or refused.

**Hallucination Rate**

A third structured prompt asks the judge to identify specific hallucinated claims — wrong numbers, wrong thresholds, invented provisions — and returns `{"hallucinated": bool, "example": str}`. This is more precise than faithfulness because it asks for specific wrong claims rather than general groundedness.

**Retrieval Hit Rate at K**

For each question in the golden set, the `answer_key` field contains a short string that must appear in a correct answer (e.g. "1.5" for overtime questions, "72 hours" for max overtime). Hit Rate at K measures what fraction of queries have this string present in the text of the top-K retrieved chunks, for K = 1, 3, 5.

The gap between Hit@1 and Hit@5 quantifies how much rank matters — if Hit@1 is 55% and Hit@5 is 90%, the correct chunk is being retrieved but not ranked first, which motivates adding a reranker.

**Citation Rate**

Fraction of answers containing at least one `[Source: ...]` pattern. Measures prompt-following and whether source attribution is working.

**Fallback Rate**

Fraction of answers containing the fallback phrase "I don't have enough information". This is a positive signal when the question genuinely cannot be answered from the retrieved context, and a negative signal when it occurs for questions the DB should be able to answer.

**Out-of-Scope Block Rate**

Fraction of the 5 off-topic test queries correctly blocked by the scope router. Ground truth labels are manually verified.

**Latency**

End-to-end wall time for `ask()`, measured with `time.perf_counter()`. Reports average, p50 (median), and p95. p95 is the operationally relevant metric for a user-facing system.

---

### LLM-as-Judge Design and Limitations

All three judge prompts (faithfulness, correctness, hallucination) use the same Llama 3.1 8B model that generated the answers (for cost reasons). This introduces **self-serving bias**: a model evaluated by itself will tend to rate its own outputs favourably compared to evaluation by an independent model. 

For a production deployment, a better approach is:
- Use a separate, stronger model as judge (GPT-4o-mini at ~$0.01 per evaluation run)
- Supplement with human annotation on a stratified sample of 50-100 questions
- Track judge-human agreement as a calibration metric

---

### Evaluation Results

Run `python eval.py` (20 qns) or `python eval.py --quick` (10 qns) 

```
Generated:
Questions: 20 | LLM Judge: enabled

Answer Quality
  Semantic Similarity (cosine, answer vs ground truth): 0.7982
  LLM Judge Faithfulness (grounded in context?): 0.8300
  LLM Judge Correctness (matches ground truth?): 0.8700
  Hallucination Rate: 10.0%

Retrieval Quality
  Hit Rate @ 1: 70.0%
  Hit Rate @ 3: 90.0%
  Hit Rate @ 5: 95.0%

Surface Metrics
  Citation Rate: 100.0%
  Fallback Rate: 0.0%
  Out-of-Scope Block Rate: 5/5

Latency
  Average: 0.83s
  p50: 0.68s
  p95: 1.08s

Out-of-scope guardrail test:
  ✓ Blocked: What is the best recipe for chicken rice?
  ✓ Blocked: What is the weather in Singapore today?
  ✓ Blocked: Can you help me write a Python script?
  ✓ Blocked: Who won the World Cup?
  ✓ Blocked: What is the price of Bitcoin?
```

---

## Ablation Studies

All ablation scripts are in `ablations/`. Run `python ablations/run_all_ablations.py` to execute all studies and generate a combined report at `ablations/ablation_report.txt`.

---

### 1. RAG vs No-RAG Baseline

**Script:** `ablations/ablation_rag_vs_baseline.py`

**Question:** Does RAG add measurable value over a bare LLM prompt?

**Method:** The same 20 golden-set questions are answered under two conditions. Condition A is bare Llama 3.1 8B with no retrieved context. Condition B is the full RAG pipeline. Both are evaluated on the same correctness and hallucination metrics. Ground truth answers are human-verified against official MOM and CPF sources.

**Why this matters:** This is the primary justification for the RAG architecture. If the bare LLM already answers correctly from its training data, RAG adds latency and complexity with no benefit. If RAG improves correctness or reduces hallucination, the overhead is justified.

**Results:**

| Condition | Correct% | Halluc% | Citation% | Refusal% |
|---|---|---|---|---|
| No RAG (bare LLM) | 75% | 15% | 0% | 0% |
| RAG pipeline | 90% | 0% | 95% | 5% |

Key finding: Rag improved correstness and decreased hallucination by 15%. 

Note: RAG failed on 2 samples: 
* How many weeks of maternity leave for a Singapore Citizen child?
  * Expected: 16 weeks; Got refusal to answer - Retrieval gap
* What are CPF contribution rates for employees aged 58?
  * Expected: 34%; Got "For employees aged 58, the CPF contribution rate is 10%  [Source: CPF CONTRIBUTIONS - Age 60-65:  Employee 10%,] - table-row retrieval problem, chunker cut across the CPF age table so the 58-year-old row landed in a different chunk than what got retrieved

---

### 2. Chunk Size

**Script:** `ablations/ablation_chunk_size.py`

**Method:** Three separate ChromaDB instances are built from the same source documents at chunk sizes 256, 512, and 1024 tokens (overlap = chunk size / 10). Each is evaluated against the same 10-question retrieval test set using Precision@5: the fraction of top-5 retrieved chunks containing the `answer_key` string for each question.

**Why 500 was chosen:** Employment Act provisions read as complete units: premise clause, condition, numerical threshold, exception. At 256 tokens, these components split across chunks, removing context. At 1024, multiple unrelated provisions merge into one chunk, reducing retrieval precision because the chunk embedding represents a mixed topic.

**Results:**

| Chunk Size | Num Chunks | Precision |
|---|---|---|
| 256 | 4131 | 0.900 |
| 512 | 1762 | 0.960 |
| 1024 | 1762 | 0.960 |

Selected: 512 

---

### 3. Top-K Retrieved Chunks

**Script:** `ablations/ablation_top_k.py`

**Method:** The full RAG pipeline is run with k = 1, 3, 5, 8 on the same 10 questions. Metrics are correctness against ground truth answers as the primary metric, and citation rate and fallback rate as secondary metrics

**Results:**
| k | Correct% | Halluc% | Citation% | Fallback% | AvgWords |
|---|---|---|---|---|---|
| 1 | 60% | 0%| 70% | 30% | 22 |
| 3 | 70% | 0%| 80% | 20% | 24 |
| 5 | 70% | 0%| 80% | 20% | 24 |
| 8 | 80% | 0%| 90% | 10% | 30 |

Selected: k=8, best correctness and citation rate 

---

### 4. Query Rewriting Robustness

**Script:** `ablations/ablation_query_rewriting.py`

**Method:** 12 deliberately imperfect queries are constructed across five categories: typos, Singlish/informal phrasing, abbreviations, code-switching, and vague questions. Each query is submitted to retrieval with and without the query understanding pipeline. Retrieval Precision is compared. 

**Categories tested:**
* Typo:  "anual leav entitlement"
* Singlish/informal: "boss never pay me salary what can i do"
* Abbreviation: "OT pay rate", "MC leave how many days"
* Code-switching: "CPF berapa percent employer pay"
* Vague: "what are my rights"

**Results:**
| Category  | Raw Precision | RW Precision | Delta |
|---|---|---|---|
| abbreviation | 0.500 | 0.800 | +0.300 |
| code-switching | 1.000 | 1.000 | +0.000 | 
| informal | 1.000 | 1.000 | +0.000 |
| singlish/informal | 0.800 | 0.867 | +0.067 |
| typo | 0.400 | 0.400 | +0.000 |
| typo + abbreviation | 0.000 | 0.400 | +0.400 |
| vague | 0.600  | 0.600 | +0.000 |
| Overall | 0.600 | 0.700 | +0.100 |

---

### 5. Out-of-Scope Detection Methods

**Script:** `ablations/ablation_scope_detection.py`

**Method:** 2 approaches (Semantic only, Semantic + LLM) are evaluated against a 40-query test set: 22 out-of-scope queries (labelled True) and 18 in-scope queries (labelled False). The out-of-scope set includes obvious cases (recipes, sports) and subtly off-topic cases that trip keyword matchers (company registration, corporate tax, HDB applications). The in-scope set includes formal, informal, and Singlish queries.

Metrics:
- **TPR** (True Positive Rate): fraction of out-of-scope queries correctly blocked  
- **TNR** (True Negative Rate): fraction of in-scope queries correctly passed  
- **FPR** (False Positive Rate): fraction of in-scope queries wrongly blocked  
- **FNR** (False Negative Rate): fraction of out-of-scope queries that slipped through 

**Results:**

| Approach | Accuracy | TPR | TNR | FPR | FNR | F1 | 
|---|---|---|---|---|---|---|---|
| Semantic only | 86% | 72% | 100% | 0% | 28% | 0.839 |  
| Tiered (Sem + LLM judge) | 97% | 94% | 100% | 0% | 6%  | 0.971 | 

Tiered received 4/36 LLM calls, and led to higher true positive and lower false negative rates. Analysing the false negative samples (out-of-scope queries that slipped through) for both cases: 
* Semantic-only FN samples: 
  * What is the best chicken rice recipe in Singapore?
  * Who won the World Cup last year?
  * Write me a poem about Singapore
  * How much does a lawyer charge per hour in Singapore?
  * What is the minimum salary to qualify for a bank loan?
* Full tiered RN samples: 
  * What is the minimum salary to qualify for a bank loan?

---

## Project Structure

```
sg-chatbot/
|
|-- app.py                    Streamlit chat interface
|-- rag_pipeline.py           Core RAG chain: retrieval, generation, guardrails
|-- ingest.py                 Web crawler, chunking, and vector DB construction
|-- eval.py                   Comprehensive evaluation suite (10 metrics)
|-- query_understanding.py    Query preprocessing: typos, Singlish, abbreviations
|-- scope_router.py           Tiered out-of-scope classifier (keyword / semantic / LLM)
|
|-- ablations/
|   |-- run_all_ablations.py          Master runner, generates ablation_report.txt
|   |-- ablation_rag_vs_baseline.py   RAG vs no-RAG comparison
|   |-- ablation_chunk_size.py        Chunk size vs retrieval precision
|   |-- ablation_top_k.py             Top-k vs answer quality
|   |-- ablation_query_rewriting.py   Query rewriting vs retrieval precision
|   |-- ablation_scope_detection.py   Scope detection method comparison
|   |-- _load_source_docs.py          Shared helper: load docs from existing DB
|
|-- data/                     PDF documents (Employment Act, etc.)
|-- db/                       ChromaDB vector index (generated by ingest.py)
|-- eval/                     Evaluation output (generated by eval.py)
|
|-- requirements.txt
|-- .env.example
|-- INSTRUCTIONS.md           Deployment guide for HuggingFace Spaces
```

---

## Setup and Running

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set environment variables**

```bash
cp .env.example .env
# Add GROQ_API_KEY from console.groq.com (free)
```

**3. Build the vector database**

```bash
python ingest.py
# Crawls MOM, CPF Board, and Workright.sg
# Expected output: 80-120 article documents, 400-600 chunks
# Runtime: approximately 3-5 minutes (rate-limited crawl)
```

**4. Run the chatbot**

```bash
streamlit run app.py
```

**5. Run evaluation**

```bash
python eval.py --quick    # 10 questions, ~8 min
python eval.py            # 20 questions, ~25 min
python eval.py --no-llm   # heuristics only, ~3 min, no API calls
```

**6. Run ablations**

```bash
python ablations/ablation_scope_detection.py   # no LLM, fast
python ablations/ablation_chunk_size.py        # no LLM, ~2 min
python ablations/run_all_ablations.py          # all studies, ~20 min
```

---

## Limitations

**Self-judge bias in evaluation.** The LLM judge uses the same model as the generator. Reported faithfulness and correctness scores are likely inflated relative to independent evaluation 

**No reranker.** Retrieved chunks are ordered by embedding cosine similarity. A cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) would re-score query-chunk pairs jointly and improve Hit@1. The current Hit@1 < Hit@5 gap is evidence that the correct chunk is being retrieved but not always ranked first

**No hybrid search.** Retrieval is dense-only. Sparse BM25 retrieval would complement it for exact-match queries involving specific legal provisions ("Section 38", "Part IV") that embedding similarity may not rank optimally

**Scope router example set size.** The semantic router is built from 45 labelled examples. A larger or more diverse example set would improve performance on edge cases, particularly queries that are adjacent to employment law but out of scope (legal fees, housing loans, business registration)

**Web scraping fragility.** The crawler relies on MOM's `data_template` meta tag to classify landing vs article pages. If MOM changes its CMS template naming, the crawler will silently fall back to the text-length heuristic, which is less reliable

**Groq rate limits.** The free tier allows approximately 30 requests per minute. Evaluation and ablation scripts include `time.sleep()` calls to respect this limit. Running multiple scripts concurrently will trigger rate limit errors.
