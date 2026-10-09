# Project Repository RAG

A Hybrid RAG application for asking questions about `Document/Dataset_project_repository.pdf`, a repository of 100 real-world projects (P001–P100).

It uses **FAISS + BM25** retrieval, **Groq** for generation, a **controlled LangGraph workflow** with answer verification and at most one retry, **SQLite** for structured project metadata, **LangSmith** for tracing and **Streamlit** for the UI.

---

## 1. Architecture

The system has two separate halves.

```
OFFLINE (run once)                                ONLINE (every question)
python ingestion/build_index.py                   streamlit run ui/app.py

PDF                                               Streamlit
 ↓ PyPDFLoader                                     ↓
group pages into projects (P001..P100)            LangGraph
 ↓                                                 ↓ analyze_query
extract project metadata                           ↓ semantic_retrieve (FAISS) ∥ keyword_retrieve (BM25)
 ↓                                                 ↓ merge_documents (+ deterministic metadata filter)
split each project (RecursiveCharacterTextSplitter) ↓ generate_answer (Groq)
 ↓ chunks inherit project metadata                 ↓ verify_answer (Groq, structured output)
Hugging Face embeddings                            ↓ valid → END
 ↓                                                 ↓ invalid → refine_query → retrieve again (max 1 retry)
FAISS  +  BM25  +  SQLite  → data/                final answer + sources
```

## 2. Why the PDF is parsed project by project

The PDF is not free text. Page 1 is a cover page. Each project then fills **exactly two pages**:

| Page | Content |
|---|---|
| Summary page | `P038 Data Mesh Implementation for Telco Data Engineering`, Problem Statement, Solution Approach, **Project Details** (Domain, Team Size, Duration, Type), Technology Stack, Expected Outcomes |
| Technical Deep Dive page | Full Technology Breakdown, Key Metrics & Business Impact, Problem Context |

Splitting the whole PDF blindly would produce chunks that start halfway through a project and contain no project ID. That caused the old `project_id = "UNKNOWN"` problem. This project uses the following order instead:

```
pages → group by project ID (page header) → extract metadata from the summary page
      → split that project's pages → every chunk inherits the same metadata
```

As a result, all 200 chunks carry `project_id`, `project_name`, `domain`, `type`, `team_size`, `duration`, `duration_weeks`, `technology_stack`, `page` and `chunk_id`. If metadata extraction fails for a project, ingestion logs it, does not index that project and never creates `UNKNOWN`.

Every chunk also starts with a short header (`Project P038: Data Mesh Implementation for Telco | Type ... | Team Size ... | Duration ...`). The second chunk of a project then still mentions its ID and name, which helps both retrievers.

Metadata is read only from the **summary page**. The deep-dive page drops units (`9` above `Duration` instead of `9 months`) and sometimes splits words (`T ool`).

## 3. Why both FAISS and BM25

| | FAISS (semantic) | BM25 (keyword) |
|---|---|---|
| Matches by | meaning (embeddings) | exact terms |
| Good for | different wording from the PDF ("cut customer attrition" → churn project) | `P038`, `Snowflake`, `Kafka`, `team size`, `Data Engineering` |
| Weak at | exact IDs and rare product names | synonyms and paraphrases |

Real example from testing: for *"What is the team size and duration of P057?"*, FAISS returned five unrelated projects, and **only BM25 found P057**. Hybrid retrieval combines both strengths. `merge_documents` removes duplicate chunks and tags each one `semantic`, `keyword` or `both`. Chunks found by both retrievers are listed first. No extra reranker is used.

## 4. Why SQLite is separate from retrieval

SQLite (`data/project.db`, table `projects`) stores **one clean row per project**. It is **not** a third retriever. It has two uses:

1. **Inspection**: browse the whole repository with the SQLite VS Code extension.
2. **Deterministic filtering**: questions like *"team size less than 5 and duration less than 3 months"* need exact comparisons, not similarity search.

The work is split between understanding the question and comparing numbers:
- **Understanding (LLM):** the single `analyze_query` LLM call fills a fixed Pydantic schema (`QueryAnalysis` in `graph/metadata_filters.py`) with the conditions in the question: team size, duration, project type and technologies. Because an LLM reads the question, typos and informal wording still work: *"atleast 5 members"*, *"more then 3 months"*, *"5+ ppl"*, *"half a year"*, *"pythn and langchian"*. A list of regular expressions would break on the next unseen typo.
- **Validation (Python):** `build_metadata_filters()` converts durations to weeks, maps technology names onto the names used in the PDF (with fuzzy matching for misspellings), and skips filters when the question names a specific project ID.
- **Comparison (Python):** `filter_projects_by_metadata()` compares the conditions against the exact values in SQLite. The LLM never decides which projects match.
- **Filtered retrieval:** when a question has conditions, FAISS and BM25 search **only the summary chunks of the matching projects** (up to 8). Each chunk is tagged at ingestion as `summary` or `technical_deep_dive`, and the summary page holds every field, including the solution approach. Without this, a question like *"solution approach of projects using Python and AWS with team size > 3"* retrieved chunks from unrelated projects, and the answer was "I don't know".

Durations are normalized at ingestion into `duration_weeks` (1 month = 52/12 weeks), so `5 weeks`, `10 weeks` and `3 months` compare correctly. The original string (`5 weeks`) is always the value that gets displayed.

Example: `team size < 5 and duration < 3 months` returns 35 projects. **P051** (2 people, 5 weeks) and **P088** (2 people, 5 weeks) are included. **P058** (3 people, 3 months) is excluded, because 3 months is not *less than* 3 months.

The complete match list goes into the prompt as an authoritative block, so the LLM never does numeric reasoning. Retrieved chunks from projects that fail the conditions are dropped. FAISS and BM25 still supply the descriptive text.

Filters are skipped when a question names a specific project (e.g. "Does P038 use Python?"), because that question is about that project.

## 5. Why indexing is offline and runs only once

Parsing the PDF and calling the embedding API on every question (or on every Streamlit rerun) would be slow and wasteful. `build_index.py` is the **only** code that builds anything:

- If `data/` already holds indexes, it stops and tells you to use `--rebuild`.
- `ui/app.py` only **loads** `data/faiss_index/`, `data/bm25.pkl` and `data/project.db`. It caches them with `st.cache_resource` and `lru_cache`, so Streamlit reruns never reload or rebuild anything.

## 6. Verification and controlled retry

```
generate_answer → verify_answer ─ valid ─────────────────────────────→ END
                                └ invalid → check_verification
                                              ├ retry_count < 1 → refine_query → FAISS ∥ BM25 → merge → generate → verify → END
                                              └ retry used up   → END
```

- `verify_answer` does **not** write a new answer. It returns `is_valid` and `verification_feedback` using structured output (a Pydantic model). It checks grounding, project IDs, names, team sizes, durations and technologies, hallucinations, whether all filter candidates were covered, and whether the question was actually answered.
- `refine_query` uses the original question, the previous search query and the verifier feedback to write a better retrieval query, and sets `retry_count = 1`.
- `check_verification` allows **at most one retry** (`MAX_RETRIES = 1` in `graph/routing.py`), so the graph cannot loop. Worst case: 2 retrieval rounds, 2 generations, 2 verifications. If the answer still fails, the UI shows it with a warning.

LLM calls per question: `analyze_query`, `generate_answer` and `verify_answer` (3 total), plus `refine_query` and one more generate/verify only on a retry. Retrieval, merging and filtering never call the LLM.

**Off-topic questions** (weather, news, general knowledge, coding help, ...) are handled by the answer prompt, with no extra node. `generate_answer` replies with a fixed message: *"This question is not related to the project repository, so I can't answer it..."*. `verify_answer` accepts that reply only when the question really is off-topic, and the UI hides the Sources table for it. A question that only mentions a repository technology, like "what is Kafka in general?", is also refused, while "Which projects use Kafka?" is answered.

`analyze_query` and `verify_answer` use Groq's **strict JSON-schema** structured output. The default tool-calling mode failed with HTTP 400 ("Tool choice is required") when the model answered an odd question in plain text.

## 7. Folder structure

```
Proj_RAG/
├── Document/
│   └── Dataset_project_repository.pdf
├── data/                       # generated by build_index.py (git-ignored)
│   ├── project.db
│   ├── faiss_index/
│   └── bm25.pkl
├── ingestion/                  # OFFLINE
│   ├── document_loader.py      # load_pdf(), group_pages_by_project()
│   ├── metadata_extractor.py   # extract_project_metadata(), normalize_duration_to_weeks()
│   ├── text_splitter.py        # split_project_into_chunks()
│   ├── embeddings.py           # get_embedding_model()
│   ├── vector_store.py         # build_vector_store(), load_vector_store()
│   ├── keyword_retriever.py    # build_keyword_retriever(), load_keyword_retriever()
│   ├── database.py             # create_database(), insert_projects(), load_project_metadata()
│   └── build_index.py          # the one-time build command
├── graph/                      # ONLINE
│   ├── state.py                # RAGState
│   ├── llm.py                  # get_llm() → ChatGroq
│   ├── metadata_filters.py     # QueryAnalysis, build_metadata_filters(), filter_projects_by_metadata()
│   ├── prompts.py              # all LLM prompts + fixed replies (NO_ANSWER, OUT_OF_SCOPE_ANSWER)
│   ├── nodes.py                # the 7 LangGraph nodes
│   ├── routing.py              # check_verification()
│   └── workflow.py             # build_graph()
├── ui/
│   └── app.py                  # Streamlit app
├── .env  ·  .env.example  ·  .gitignore  ·  requirements.txt  ·  README.md
└── myvenv/
```

`graph/metadata_filters.py` is the one addition to the suggested layout. It keeps the deterministic filter logic out of `nodes.py`.

---

## Setup

### Environment variables (`.env`)

Copy the template and fill in your keys: `copy .env.example .env` (Windows) or `cp .env.example .env`.

```
GROQ_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
LANGSMITH_ENDPOINT=          # optional
GROQ_MODEL=                  # optional, default: openai/gpt-oss-120b
```

### Install

```bash
# Windows
myvenv\Scripts\activate
pip install -r requirements.txt
```

### Build the indexes (once)

```bash
python ingestion/build_index.py            # builds data/ — stops if it already exists
python ingestion/build_index.py --rebuild  # explicit rebuild
```

Expected output: `Loaded 201 pages, found 100 projects.` → `Created 200 chunks from 100 projects.` → `Done.`

### Start the app

```bash
streamlit run ui/app.py
```

---

## Main functions

| Function | File | What it does |
|---|---|---|
| `load_pdf()` | document_loader | PyPDFLoader, one Document per page |
| `group_pages_by_project()` | document_loader | Uses the `P0xx` page header to group the summary and deep-dive pages per project; skips the cover |
| `extract_project_metadata()` | metadata_extractor | Reads ID, name, domain, type, team size, duration and technology stack from the summary page; raises if a field is missing |
| `normalize_duration_to_weeks()` | metadata_extractor | `"3 months"` → `13.0`, `"5 weeks"` → `5.0` |
| `split_project_into_chunks()` | text_splitter | Splits each page of one project (1000 / 100) and attaches metadata plus a header |
| `build_vector_store()` / `load_vector_store()` | vector_store | Embeds in batches of 32 and saves FAISS; the loader never rebuilds |
| `build_keyword_retriever()` / `load_keyword_retriever()` | keyword_retriever | BM25 with a lowercase alphanumeric tokenizer, pickled |
| `create_database()` / `insert_projects()` | database | One SQLite row per project |
| `QueryAnalysis` | metadata_filters | Pydantic schema the LLM fills: search query + team size / duration / type / technology conditions |
| `build_metadata_filters()` | metadata_filters | Validates the LLM's conditions (weeks conversion, technology-name matching) |
| `filter_projects_by_metadata()` | metadata_filters | Compares those conditions in Python against the SQLite values |
| `analyze_query()` | nodes | One structured-output LLM call: retrieval query + conditions (handles typos and informal wording) |
| `semantic_retrieve()` | nodes | FAISS `similarity_search(k=5)` |
| `keyword_retrieve()` | nodes | BM25 `invoke()` (k=5) |
| `merge_documents()` | nodes | De-duplicates by chunk, tags the retrieval method, applies metadata filters |
| `generate_answer()` | nodes | Groq answer from the context only |
| `verify_answer()` | nodes | Groq structured verdict (`is_valid`, `feedback`) |
| `refine_query()` | nodes | Better query from the verifier feedback; increments `retry_count` |
| `check_verification()` | routing | `"end"` or `"retry"` |
| `build_graph()` | workflow | Assembles and compiles the StateGraph |

## LangGraph execution

1. `analyze_query` sets `search_query`, `metadata_filters` and `retry_count = 0`.
2. `semantic_retrieve` and `keyword_retrieve` run **in parallel** (two edges out of `analyze_query`).
3. `merge_documents` waits for both (`add_edge([...], "merge_documents")`).
4. `generate_answer`, then `verify_answer`.
5. `check_verification` routes to `END`, or once to `refine_query`, which fans out to both retrievers again.

## Sample questions

| Question | What happens |
|---|---|
| What technologies are used in P038? | BM25 and FAISS both find P038 → GCP Dataplex, BigQuery, dbt, Apache Kafka, Terraform, Python |
| Tell me about P088. | Full project summary |
| What is the team size and duration of P057? | Only BM25 finds P057 → 3 people, 3 months |
| What is the solution approach of P090? | FHIR R4 integration layer, master patient index, HL7 v2 → FHIR converter |
| Which projects have a team size less than 5 and duration less than 3 months? | Filter `team_size < 5`, `duration_weeks < 13` → 35 projects, including P051 and P088, excluding P058 |
| Which projects use Python and Snowflake? | Technology filter (all) → 17 projects |
| Which Data Engineering projects use Python? | Type + technology filter → 17 projects |
| Which projects are related to NLP / GenAI? | Type filter → 34 projects |
| What technologies are used across the Data Engineering projects? | Type filter + exact technology tally across the 22 projects |
| Which projects use Kafka or Spark? | Technology filter (any) |

## Inspect SQLite

1. Install the **SQLite Viewer** (or **SQLite**) extension in VS Code.
2. Open `data/project.db` and open the `projects` table.

Or from Python:

```bash
python -c "import sqlite3; c=sqlite3.connect('data/project.db'); [print(r) for r in c.execute('SELECT project_id, project_name, type, team_size, duration, duration_weeks FROM projects WHERE team_size < 5 AND duration_weeks < 13')]"
```

## Inspect the LangSmith trace

1. Make sure `.env` has `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`.
2. Ask a question in the app.
3. Open [smith.langchain.com](https://smith.langchain.com) → **Tracing Projects** → your project → the run **"Project Repository RAG"**.
4. Expand it to see `analyze_query`, `semantic_retrieve`, `keyword_retrieve` (with the BM25Retriever run), `merge_documents`, `generate_answer`, `verify_answer`, `check_verification` and, on a retry, `refine_query` with the second pass. Each ChatGroq call shows its prompt, output and token usage.

## Notes and limits

- **Groq free tier:** `openai/gpt-oss-120b` allows about 8,000 tokens per minute. Long list answers (20–35 projects) use about 5k tokens each for generation and verification, so the client waits and retries on HTTP 429. Those questions can take 40–100 seconds. A paid tier, or a different `GROQ_MODEL`, removes the wait. (`llama-3.3-70b-versatile` is no longer available on this account.)
- **Embeddings:** all-MiniLM-L6-v2 runs **locally** with FastEmbed (ONNX Runtime). It is downloaded from Hugging Face once and cached, so it needs no API key, no credits and no PyTorch. An earlier version used the Hugging Face Inference API, which stopped working when the account's free credits ran out (HTTP 402).
- **Windows note:** `onnxruntime` is pinned to 1.18.1. Newer builds (and PyTorch) crash on PCs with an older Microsoft Visual C++ Runtime (14.29 here). If you install the latest [VC++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe), you can remove that pin.
- **Supported filter wording:** team size and duration with comparisons such as *less than, under, at most, more than, at least, exactly, <, >*; project type; and technology names that appear in the PDF. Use *or* between technologies to match any of them; the default is all. One comparison per field (no "between 3 and 5").
- **Tech-name cleanup:** pypdf inserts a space before hyphens (`GPT -4`). Ingestion removes it so names match the PDF (`GPT-4`).
