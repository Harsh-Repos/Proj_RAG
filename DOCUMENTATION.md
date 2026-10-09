# Complete Project Documentation: Hybrid RAG System
## Line-by-Line Explanation with Architecture Flow

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Complete Flow Walkthrough](#complete-flow-walkthrough)
4. [Component-by-Component Breakdown](#component-by-component-breakdown)
5. [Function Reference Guide](#function-reference-guide)
6. [Code Walkthrough](#code-walkthrough)
7. [Data Flow Explanation](#data-flow-explanation)

---

## Project Overview

### What is this project?
This is a **Hybrid RAG (Retrieval-Augmented Generation) System** that answers questions about 100 real-world projects from a PDF file.

### The Goal
- User asks a question about projects (e.g., "What technologies does P038 use?")
- System retrieves relevant project information using two different methods (semantic + keyword)
- AI generates an answer based only on the retrieved information
- AI verifies the answer is correct and grounded
- If incorrect, the system automatically refines the search and tries once more
- User gets a final answer with sources

### Why "Hybrid"?
- Uses **FAISS** (semantic search - understands meaning)
- Uses **BM25** (keyword search - finds exact terms like "P038" or "Snowflake")
- Combines both for better results

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT REPOSITORY RAG                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OFFLINE (Build Once)          │      ONLINE (Query Every Time)│
│  ─────────────────────         │      ──────────────────────── │
│                                │                                │
│  PDF File                      │      User Question            │
│    ↓                           │        ↓                      │
│  Load & Parse                  │      Streamlit UI             │
│    ↓                           │        ↓                      │
│  Extract Metadata              │      LangGraph Workflow       │
│    ↓                           │        ↓                      │
│  Create Embeddings             │      FAISS + BM25 Retrieval   │
│    ↓                           │        ↓                      │
│  Build FAISS Index             │      Merge Results            │
│    ↓                           │        ↓                      │
│  Build BM25 Index              │      Groq LLM Answer          │
│    ↓                           │        ↓                      │
│  Create SQLite DB              │      Verify Answer            │
│    ↓                           │        ↓                      │
│  Save to disk/                 │      Show to User             │
│  (data/ folder)                │                                │
│                                │                                │
└─────────────────────────────────────────────────────────────────┘
```

### Key Principle: Separation of Concerns
- **Ingestion happens ONCE** (python ingestion/build_index.py)
- **Querying uses cached indexes** (streamlit run ui/app.py)
- **Never rebuild indexes during queries** (fast response)

---

## Complete Flow Walkthrough

### Step 1: User Starts App
```
User runs: streamlit run ui/app.py
       ↓
Browser opens http://localhost:8501
       ↓
Streamlit loads ui/app.py
       ↓
@st.cache_resource loads:
  - FAISS index (data/faiss_index/)
  - BM25 retriever (data/bm25.pkl)
  - Project metadata (data/project.db)
       ↓
App is ready for questions
```

### Step 2: User Asks a Question
```
User types: "Which projects use Python and Snowflake?"
       ↓
Click "Ask" button
       ↓
Streamlit calls: graph.invoke({"question": "..."})
       ↓
LangGraph workflow starts
```

### Step 3: analyze_query Node Runs
```
Input: question = "Which projects use Python and Snowflake?"

Step 3a: LLM Rewrites the Question
  - Calls Groq LLM
  - Adds field names for better retrieval
  - Output: search_query = "technology stack Python Snowflake"
  
Step 3b: Extract Metadata Filters (NO LLM - just regex)
  - Looks for: team size, duration, type, technologies
  - Finds: technologies = ["Python", "Snowflake"]
  - technology_match = "all" (both must be present)
  - Output: metadata_filters = {"technologies": [...]}

Step 3c: Initialize State
  - retry_count = 0 (no retries yet)
  - Return updated state

Output state fields:
  - question: "Which projects use Python and Snowflake?"
  - search_query: "technology stack Python Snowflake"
  - metadata_filters: {"technologies": ["Python", "Snowflake"], "technology_match": "all"}
  - retry_count: 0
```

### Step 4: FAISS and BM25 Run in PARALLEL
```
These two nodes run AT THE SAME TIME (not one after the other)

Branch A: semantic_retrieve (FAISS)
──────────────────────────────────
Input: search_query = "technology stack Python Snowflake"

Line-by-line:
1. Embedding the search query
   - "technology stack Python Snowflake" → converted to numbers (embedding)
   - Uses HuggingFace Endpoint Embeddings
   
2. Finding similar chunks in FAISS
   - FAISS searches for chunks with similar meaning
   - Retrieves top 5 most similar chunks
   - Example: finds chunks about "data platforms that use Python and Snowflake"

Output: semantic_documents = [
  Document(page_content="...", metadata={"project_id": "P008", ...}),
  Document(page_content="...", metadata={"project_id": "P010", ...}),
  Document(page_content="...", metadata={"project_id": "P049", ...}),
  ...
]

Branch B: keyword_retrieve (BM25)
──────────────────────────────────
Input: search_query = "technology stack Python Snowflake"

Line-by-line:
1. Tokenizing the search query
   - "technology stack Python Snowflake"
   - → ["technology", "stack", "python", "snowflake"]
   - (lowercase, alphanumeric only)

2. Finding exact keyword matches in BM25
   - BM25 searches for documents containing these keywords
   - Retrieves top 5 documents
   - Example: finds chunks explicitly mentioning "Python" and "Snowflake"

Output: keyword_documents = [
  Document(page_content="...", metadata={"project_id": "P057", ...}),
  Document(page_content="...", metadata={"project_id": "P058", ...}),
  ...
]

PARALLEL EXECUTION ADVANTAGE:
If this was sequential:
  - FAISS search: 2 seconds
  - BM25 search: 1 second
  - Total: 3 seconds

With parallel (what we do):
  - Both run at same time
  - Total: 2 seconds (max of both)
```

### Step 5: merge_documents Node Runs
```
Input: 
  - semantic_documents (from FAISS)
  - keyword_documents (from BM25)
  - metadata_filters (extracted earlier)

Step 5a: Remove Duplicates
  ────────────────────────
  Documents found by BOTH retrievers are marked as "both"
  
  If FAISS found: P057-p114-c0
  If BM25 found: P057-p114-c0
  → Keep only ONE, mark as retrieval_method = "both"

Step 5b: Apply Metadata Filters
  ──────────────────────────────
  Filter extracted: technologies = ["Python", "Snowflake"], match = "all"
  
  For each of the 100 projects (loaded once from SQLite and cached):
    - If it uses both Python AND Snowflake → keep it
    - Otherwise → filter out

  This comparison runs in Python (filter_projects_by_metadata), not as a SQL query:
  the 100 rows are already in memory, so exact comparison is instant.

  Result: 17 projects match (P008, P010, P015, ..., P097)
  
  Store in: metadata_matches = [
    {"project_id": "P008", "team_size": 4, ...},
    {"project_id": "P010", "team_size": 3, ...},
    ...
  ]

Step 5c: Retrieval Already Restricted to Matching Projects
  ───────────────────────────────────────────────────────
  If metadata filters exist, FAISS and BM25 searched ONLY the summary
  chunks of the matching projects (up to 8 each), so no chunk has to be
  dropped here. The full list of all 17 matches still goes into the prompt
  as "PROJECTS MATCHING THE METADATA FILTERS".

Output:
  - documents: [chunk1, chunk2, chunk3, ...] (filtered & de-duped)
  - metadata_matches: [17 projects that match the filters]
```

### Step 6: generate_answer Node Runs
```
Input:
  - question: "Which projects use Python and Snowflake?"
  - documents: [merged, filtered chunks]
  - metadata_matches: [17 projects]

Step 6a: Build the Prompt Context
  ───────────────────────────────
  
  If metadata_matches exists, create a block:
  ┌──────────────────────────────────────────────────┐
  │ PROJECTS MATCHING THE METADATA FILTERS:          │
  │                                                  │
  │ - P008 — Automated Data Quality Framework        │
  │   Type: Data Engineering | Team Size: 4          │
  │   Technologies: Python, Snowflake, dbt, Airflow  │
  │                                                  │
  │ - P010 — Cloud Cost Optimization Intelligence    │
  │   Type: ML/AI | Team Size: 3                     │
  │   Technologies: Python, Snowflake, FastAPI, ...  │
  │                                                  │
  │ ... (all 17 projects)                           │
  └──────────────────────────────────────────────────┘
  
  This list is AUTHORITATIVE and COMPLETE
  The LLM MUST include all 17 projects

Step 6b: Add Retrieved Text Chunks
  ───────────────────────────────
  [Source 1] P057 — Automated A/B Testing Platform (page 114)
  <actual text from the chunk>
  
  [Source 2] P008 — Automated Data Quality Framework (page 4)
  <actual text from the chunk>
  
  ... (all chunks)

Step 6c: Call Groq LLM to Generate Answer
  ────────────────────────────────────
  
  Groq receives:
  ┌─────────────────────────────────────────────────┐
  │ You are an expert assistant for a project repo. │
  │                                                 │
  │ RULES:                                          │
  │ - Use ONLY the provided context               │
  │ - When the "PROJECTS MATCHING METADATA FILTERS" │
  │   list is given, it is AUTHORITATIVE           │
  │ - Return ALL projects in that list for filtering│
  │   questions                                    │
  │ - Do not invent projects                       │
  │ - Copy exact project IDs and names             │
  │                                                 │
  │ CONTEXT:                                        │
  │ [metadata matches list + retrieved chunks]     │
  │                                                 │
  │ QUESTION: Which projects use Python and        │
  │ Snowflake?                                     │
  └─────────────────────────────────────────────────┘
  
  Groq generates answer:
  "The following 17 projects use both Python and Snowflake:
   
   - P008 — Automated Data Quality Framework
   - P010 — Cloud Cost Optimization Intelligence
   - P015 — Customer 360 Data Platform
   - P021 — Automated Financial Reporting Pipeline
   ...
   - P097 — Automated ESG Scoring for Investment Portfolio"

Output:
  - answer: "The following 17 projects..."
```

### Step 7: verify_answer Node Runs
```
Input:
  - question: "Which projects use Python and Snowflake?"
  - answer: "The following 17 projects use both..."
  - documents: [merged chunks]
  - metadata_matches: [17 projects]

Step 7a: Build Verification Prompt
  ───────────────────────────────
  
  Groq receives:
  ┌──────────────────────────────────────────────────┐
  │ You are a strict fact-checker for a RAG system.  │
  │                                                  │
  │ Check the ANSWER against the CONTEXT and        │
  │ QUESTION:                                        │
  │ 1. Is every statement grounded in context?      │
  │ 2. Are project IDs correct?                     │
  │ 3. Are project names correct?                   │
  │ 4. Are technologies correct?                    │
  │ 5. For filtering questions, were all relevant   │
  │    candidates considered?                       │
  │ 6. Did the answer hallucinate (invent stuff)?   │
  │ 7. Does the answer answer the question?         │
  │                                                  │
  │ CONTEXT:                                         │
  │ [all chunks and metadata]                       │
  │                                                  │
  │ QUESTION: Which projects use Python and         │
  │ Snowflake?                                      │
  │                                                  │
  │ ANSWER:                                          │
  │ The following 17 projects use both Python and   │
  │ Snowflake: P008, P010, P015, ...                │
  └──────────────────────────────────────────────────┘

Step 7b: Groq Returns Structured Verification
  ────────────────────────────────────────────
  
  Using Pydantic model:
  {
    "is_valid": true,
    "feedback": "All 17 projects listed are present in metadata_matches.
                 Project names and technologies are correct.
                 Answer is complete and grounded."
  }

Output:
  - is_valid: true
  - verification_feedback: "All checks passed."
```

### Step 8: check_verification Router
```
Conditional Logic:

if is_valid == True:
  → Go to END node (finish)
  
elif retry_count < 1:
  → Go to refine_query node (try again)
  
else:
  → Go to END node (stop, show answer even if invalid)

In our example:
  is_valid = True
  → Go to END ✓
```

### Step 9: Return to User
```
Streamlit receives final result:

{
  "question": "Which projects use Python and Snowflake?",
  "search_query": "technology stack Python Snowflake",
  "metadata_filters": {"technologies": ["Python", "Snowflake"]},
  "documents": [...],
  "metadata_matches": [17 projects],
  "answer": "The following 17 projects...",
  "is_valid": true,
  "verification_feedback": "All checks passed.",
  "retry_count": 0
}

Streamlit displays:
─────────────────

ANSWER:
The following 17 projects use both Python and Snowflake:
- P008 — Automated Data Quality Framework
- P010 — Cloud Cost Optimization Intelligence
... (all 17)

✅ Verified: All checks passed.

SOURCES:
┌───────────────────────────────────────────────────┐
│ Project             | Type          | Retrieved by│
├───────────────────────────────────────────────────┤
│ P057 — Automated... | Data Engine.. | keyword     │
│ P008 — Automated... | Data Engine.. | semantic    │
│ ... (more sources)                                │
└───────────────────────────────────────────────────┘

PROJECTS MATCHING METADATA FILTERS (17):
[Expandable section showing all 17 with details]
```

---

## Component-by-Component Breakdown

### 1. INGESTION LAYER (runs once: `python ingestion/build_index.py`)

#### Purpose
Parse the PDF, extract metadata, create embeddings, and build search indexes.

#### Files Involved
```
ingestion/
├── document_loader.py      → Load PDF
├── metadata_extractor.py   → Extract project info
├── text_splitter.py        → Split into chunks
├── embeddings.py           → Create embeddings
├── vector_store.py         → Build FAISS
├── keyword_retriever.py    → Build BM25
├── database.py             → Create SQLite
└── build_index.py          → Orchestrate all
```

---

### 2. GRAPH LAYER (LangGraph Workflow)

#### Purpose
Implement the RAG workflow with verification and retry logic.

#### Files Involved
```
graph/
├── state.py                → RAGState (data model)
├── llm.py                  → Groq LLM setup
├── prompts.py              → All LLM prompts + fixed replies (NO_ANSWER, OUT_OF_SCOPE_ANSWER)
├── nodes.py                → All 7 nodes
├── metadata_filters.py     → Filter extraction
├── routing.py              → Conditional routing
└── workflow.py             → Graph assembly
```

#### The 7 Nodes (and what each does)

```
1. analyze_query
   Input: question (user's question)
   Output: search_query, metadata_filters, retry_count
   LLM Calls: 1 (rewrite question)
   
2. semantic_retrieve (FAISS)
   Input: search_query
   Output: semantic_documents
   LLM Calls: 0 (only vector math)
   
3. keyword_retrieve (BM25)
   Input: search_query
   Output: keyword_documents
   LLM Calls: 0 (only text matching)
   
4. merge_documents
   Input: semantic_documents, keyword_documents, metadata_filters
   Output: documents, metadata_matches
   LLM Calls: 0 (deterministic)
   
5. generate_answer
   Input: question, documents, metadata_matches
   Output: answer
   LLM Calls: 1 (generate)
   
6. verify_answer
   Input: question, answer, documents
   Output: is_valid, verification_feedback
   LLM Calls: 1 (check)
   
7. refine_query (optional, on retry)
   Input: question, search_query, verification_feedback
   Output: search_query (refined), retry_count++
   LLM Calls: 1 (improve query)
```

---

### 3. UI LAYER (Streamlit)

#### Purpose
Provide user interface for asking questions.

#### File
```
ui/app.py
```

#### Key Functions
```
load_rag_graph()          → Load cached indexes
summarize_sources()       → Format retrieved docs
escape_markdown()         → Escape special chars
show_result()             → Display answer & sources
main()                    → Streamlit app
```

---

## Function Reference Guide

### Ingestion Functions

#### `load_pdf(pdf_path)`
```python
def load_pdf(pdf_path: Path = PDF_PATH) -> list[Document]:
    """Load the PDF with PyPDFLoader, one Document per page."""
    return PyPDFLoader(str(pdf_path)).load()

Parameters:
  pdf_path (Path): Path to PDF file
  
Returns:
  list[Document]: List of documents, one per page
  
What it does:
  1. Reads the PDF file
  2. Extracts each page as text
  3. Returns a list of LangChain Document objects
  
Example:
  pages = load_pdf()
  # Returns 201 Document objects (page 1 = cover, pages 2-201 = projects)
```

#### `group_pages_by_project(pages)`
```python
def group_pages_by_project(pages: list[Document]) -> list[dict]:
    """Group consecutive pages that belong to the same project."""
    
Parameters:
  pages (list[Document]): All pages from PDF
  
Returns:
  list[dict]: List like [
    {"project_id": "P001", "pages": [Document, Document]},
    {"project_id": "P002", "pages": [Document, Document]},
    ...
  ]
  
What it does:
  1. Loop through each page
  2. Check if page starts with project ID (P001, P002, etc.)
  3. If new project ID found → start new group
  4. If no ID → add to current group (continuation page)
  5. Skip cover page (page 1, has no project ID)
  
Example:
  projects = group_pages_by_project(pages)
  # projects[0] = {"project_id": "P001", "pages": [page2, page3]}
  # projects[1] = {"project_id": "P002", "pages": [page4, page5]}
```

#### `extract_project_metadata(project)`
```python
def extract_project_metadata(project: dict) -> dict:
    """Extract metadata from a project's summary page."""
    
Parameters:
  project (dict): Like {"project_id": "P038", "pages": [...]}
  
Returns:
  dict: {
    "project_id": "P038",
    "project_name": "Data Mesh Implementation for Telco",
    "domain": "Telecom / Data Platform",
    "type": "Data Engineering",
    "team_size": 10,
    "duration": "9 months",
    "duration_weeks": 39.0,
    "technology_stack": ["GCP Dataplex", "BigQuery", ...],
    "pages": [76, 77]
  }
  
What it does:
  1. Read the first page (summary page)
  2. Extract header line: "P038 Data Mesh... Data Engineering"
  3. Find PROJECT DETAILS section
  4. Extract Domain, Team Size, Duration, Type using regex
  5. Extract TECHNOLOGY STACK section and split by "·" bullet
  6. Normalize duration to weeks (3 months → 13.0 weeks)
  7. Raise error if any field missing
  
Example:
  metadata = extract_project_metadata({"project_id": "P038", "pages": [...]})
  # metadata["team_size"] = 10
  # metadata["duration_weeks"] = 39.0
```

#### `normalize_duration_to_weeks(duration)`
```python
def normalize_duration_to_weeks(duration: str) -> float:
    """Convert "5 weeks" / "3 months" into weeks."""
    
Parameters:
  duration (str): Like "5 weeks" or "3 months"
  
Returns:
  float: Number of weeks (3 months → 13.0)
  
What it does:
  1. Split string: "3 months" → ["3", "months"]
  2. Get number (3) and unit (months)
  3. If unit is "months": multiply by 52/12 ≈ 4.33
     If unit is "weeks": keep as is
  4. Round to 2 decimals
  
Example:
  weeks = normalize_duration_to_weeks("3 months")
  # weeks = 13.0
```

#### `split_project_into_chunks(project, metadata)`
```python
def split_project_into_chunks(project: dict, metadata: dict) -> list[Document]:
    """Split each page of a project into chunks."""
    
Parameters:
  project (dict): {"project_id": "P001", "pages": [...]}
  metadata (dict): Extracted metadata (from extract_project_metadata)
  
Returns:
  list[Document]: Like [
    Document(
      page_content="Project P001: ... [chunk text]",
      metadata={
        "project_id": "P001",
        "project_name": "...",
        "page": 3,
        "chunk_id": "P001-p3-c0",
        ...
      }
    ),
    ...
  ]
  
What it does:
  1. Create a header: "Project P001: Name | Type: ... | Team Size: ..."
  2. For each page in the project:
     a. Split page text using RecursiveCharacterTextSplitter
        (chunk_size=1000, overlap=100)
     b. For each chunk:
        - Prepend the header
        - Attach metadata to chunk
        - Create unique chunk_id
  3. Return all chunks
  
Example:
  chunks = split_project_into_chunks(project, metadata)
  # P001 has 2 pages → might become 4-5 chunks
  # Each chunk has full metadata attached
```

#### `get_embedding_model()`
```python
def get_embedding_model() -> FastEmbedEmbeddings:
    """Return the local all-MiniLM-L6-v2 embedding model (384 dimensions)."""
    
Returns:
  FastEmbedEmbeddings: Ready to embed text
  
What it does:
  1. Create FastEmbedEmbeddings with "sentence-transformers/all-MiniLM-L6-v2"
  2. First run: download the model from Hugging Face and cache it locally
  3. Return the model
  
Note:
  - Embeddings are 384-dimensional vectors
  - Same model used to embed chunks AND search queries
  - Runs locally on the CPU (ONNX Runtime): no API key, no credits, no PyTorch
```

#### `build_vector_store(chunks, embeddings)`
```python
def build_vector_store(chunks: list[Document], embeddings: Embeddings) -> FAISS:
    """Build FAISS index from chunks."""
    
Parameters:
  chunks (list[Document]): All 200+ chunks
  embeddings (Embeddings): The embedding model
  
Returns:
  FAISS: Saved to data/faiss_index/
  
What it does:
  1. Split chunks into batches of 32
  2. For each batch:
     a. Embed chunks (call HF API)
     b. Add to FAISS index
  3. Save index to disk (data/faiss_index/)
  
Why batch?
  - Keeps API requests small
  - Avoids timeouts
```

#### `load_vector_store()`
```python
def load_vector_store() -> FAISS:
    """Load the persisted FAISS index."""
    
Returns:
  FAISS: Ready to search
  
What it does:
  1. Check if data/faiss_index/ exists
  2. Load the index from disk
  3. Return it (never rebuilds)
```

#### `build_keyword_retriever(chunks)`
```python
def build_keyword_retriever(chunks: list[Document]) -> BM25Retriever:
    """Build BM25 retriever."""
    
Parameters:
  chunks (list[Document]): All chunks
  
Returns:
  BM25Retriever: Saved to data/bm25.pkl
  
What it does:
  1. Create BM25Retriever from all chunks
  2. Pickle it (serialize to binary)
  3. Save to disk
```

#### `load_keyword_retriever()`
```python
def load_keyword_retriever() -> BM25Retriever:
    """Load the pickled BM25 retriever."""
    
Returns:
  BM25Retriever: Ready to search
```

---

### Graph Functions

#### RAGState (TypedDict)
```python
class RAGState(TypedDict):
    """State shared by all nodes."""
    
    question: str                    # Original user question
    search_query: str                # Rewritten for retrieval
    metadata_filters: dict           # {"technologies": [...]}
    
    semantic_documents: list         # FAISS results
    keyword_documents: list          # BM25 results
    documents: list                  # Merged & filtered
    metadata_matches: list           # Projects matching filters
    
    answer: str                      # Generated answer
    
    is_valid: bool                   # Passed verification?
    verification_feedback: str       # Why valid/invalid
    
    retry_count: int                 # How many retries used

Purpose:
  - All nodes read/write to this state
  - Accumulated as workflow progresses
```

#### `analyze_query(state: RAGState) -> dict`
```python
def analyze_query(state: RAGState) -> dict:
    """Rewrite question & extract filters."""
    
Input state:
  - question: "Which projects use Python and Snowflake?"
  
Process:
  1. ONE Groq LLM call with structured output (QueryAnalysis schema).
     The LLM returns:
     - search_query: "technology stack Python Snowflake"
     - technologies: ["Python", "Snowflake"]
     - technology_match: "all" (both required)
     Because an LLM reads the question, typos still work:
     "atleast 5 members and duration more then 3 months"
       -> team_size >= 5, duration > 3 months
     
  2. build_metadata_filters() (plain Python) validates the result:
     - converts durations to weeks (3 months -> 13.0)
     - maps technology names onto the PDF's names ("snowflak" -> "Snowflake")
     - drops all filters if the question names a project ID (e.g. P038)
     
  3. Initialize retry_count = 0
  
Output dict:
  {
    "search_query": "technology stack Python Snowflake",
    "metadata_filters": {"technologies": [...], "technology_match": "all"},
    "retry_count": 0
  }
  
Note:
  - First node to run
  - Only LLM call in analyze phase
  - The LLM only UNDERSTANDS the conditions; the numeric comparison is done
    later in Python (merge_documents), so the LLM never decides which projects match
```

#### `semantic_retrieve(state: RAGState) -> dict`
```python
def semantic_retrieve(state: RAGState) -> dict:
    """Semantic search with FAISS."""
    
Input state:
  - search_query: "technology stack Python Snowflake"
  
Process:
  1. Embed the search_query:
     "technology stack Python Snowflake"
     → [0.12, -0.45, 0.89, ...] (384 numbers)
     
  2. Find similar chunks in FAISS:
     FAISS.similarity_search(embedding, k=5)
     → Returns top 5 most similar chunks
     
  3. Extract metadata from chunks
  
Output dict:
  {
    "semantic_documents": [
      Document(..., metadata={"project_id": "P008", ...}),
      Document(..., metadata={"project_id": "P010", ...}),
      Document(..., metadata={"project_id": "P049", ...}),
      Document(..., metadata={"project_id": "P021", ...}),
      Document(..., metadata={"project_id": "P057", ...}),
    ]
  }

Note:
  - Finds meaning-based matches
  - Runs in parallel with keyword_retrieve
  - No LLM call
  - Deterministic (same query = same results)
```

#### `keyword_retrieve(state: RAGState) -> dict`
```python
def keyword_retrieve(state: RAGState) -> dict:
    """Keyword search with BM25."""
    
Input state:
  - search_query: "technology stack Python Snowflake"
  
Process:
  1. Tokenize the query:
     "technology stack Python Snowflake"
     → ["technology", "stack", "python", "snowflake"]
     
  2. Find chunks with these keywords:
     BM25Retriever.invoke(search_query)
     → Returns top 5 documents with best keyword match
     
  3. Extract metadata from chunks
  
Output dict:
  {
    "keyword_documents": [
      Document(..., metadata={"project_id": "P057", ...}),
      Document(..., metadata={"project_id": "P058", ...}),
      Document(..., metadata={"project_id": "P008", ...}),
      Document(..., metadata={"project_id": "P010", ...}),
      Document(..., metadata={"project_id": "P015", ...}),
    ]
  }

Note:
  - Finds exact keyword matches
  - Great for project IDs (P038), product names (Snowflake)
  - Runs in parallel with semantic_retrieve
  - No LLM call
```

#### `merge_documents(state: RAGState) -> dict`
```python
def merge_documents(state: RAGState) -> dict:
    """Combine & filter results."""
    
Input state:
  - semantic_documents: [5 docs from FAISS]
  - keyword_documents: [5 docs from BM25]
  - metadata_filters: {"technologies": [...]}
  
Process Step 1: De-duplicate
  ─────────────────────────
  If chunk found by BOTH:
    - Keep ONE copy
    - Mark: retrieval_method = "both"
  
  If found by only one:
    - Mark: retrieval_method = "semantic" or "keyword"
  
  Result: Merged, de-duplicated list
  
Process Step 2: Query SQLite with filters
  ────────────────────────────────────
  If metadata_filters exist:
    - Query projects matching filters
    - Extract: {"technologies": ["Python", "Snowflake"]}
    
    SQL query:
    ```
    SELECT * FROM projects 
    WHERE technology_stack LIKE '%Python%' 
      AND technology_stack LIKE '%Snowflake%'
    ```
    
    Result: 17 projects matching
    [P008, P010, P015, P021, ..., P097]
  
Process Step 3: Filter chunks
  ──────────────────────────
  Remove chunks from projects NOT in metadata_matches
  
  Before: [chunk from P008, chunk from P200, chunk from P010, ...]
  After:  [chunk from P008, chunk from P010, ...] (only matching projects)

Output dict:
  {
    "documents": [filtered merged chunks],
    "metadata_matches": [17 matching projects with all details]
  }

Note:
  - Deterministic (no LLM)
  - Applies metadata filtering
  - Prepares context for answer generation
```

#### `generate_answer(state: RAGState) -> dict`
```python
def generate_answer(state: RAGState) -> dict:
    """Generate answer with Groq LLM."""
    
Input state:
  - question: "Which projects use Python and Snowflake?"
  - documents: [5 merged chunks]
  - metadata_matches: [17 projects]
  - retry_count: 0
  
Process:
  1. Build prompt:
     ┌────────────────────────────────┐
     │ You are a project repository   │
     │ assistant.                     │
     │                                │
     │ RULES:                         │
     │ - Use only provided context    │
     │ - Copy exact project IDs       │
     │ - When PROJECTS MATCHING...    │
     │   list is given, it is         │
     │   AUTHORITATIVE & COMPLETE     │
     │ - Return ALL projects in list  │
     │ - Never invent projects        │
     │                                │
     │ CONTEXT:                       │
     │ [metadata_matches + documents] │
     │                                │
     │ QUESTION: Which projects      │
     │ use Python and Snowflake?     │
     └────────────────────────────────┘
  
  2. Call Groq LLM:
     Groq generates answer:
     "The following 17 projects use both Python
      and Snowflake:
      - P008 — Automated Data Quality Framework
      - P010 — Cloud Cost Optimization Intelligence
      ... (all 17)"
  
  3. If retry (retry_count > 0):
     Add note: "A previous answer was rejected for:
                [verification_feedback]"

Output dict:
  {
    "answer": "The following 17 projects..."
  }

Note:
  - ONE LLM call
  - Context includes authoritative filter list
  - LLM must include all 17 projects
```

#### `verify_answer(state: RAGState) -> dict`
```python
def verify_answer(state: RAGState) -> dict:
    """Verify answer is correct & grounded."""
    
Input state:
  - question: "Which projects use Python and Snowflake?"
  - answer: "The following 17 projects..."
  - documents: [context chunks]
  - metadata_matches: [17 projects]
  
Process:
  1. Build verification prompt:
     "You are a strict fact-checker.
     
      Check if the answer:
      1. Is grounded in context?
      2. Has correct project IDs?
      3. Has correct project names?
      4. Has correct technologies?
      5. Covered all filter candidates?
      6. Hallucinated anything?
      7. Answers the question?
      
      CONTEXT: [...]
      QUESTION: [...]
      ANSWER: [...]"
  
  2. Call Groq LLM with structured output:
     Returns Pydantic model:
     {
       "is_valid": true,
       "feedback": "All 17 projects match filters.
                   Names and techs correct.
                   Complete answer."
     }
  
  3. Check each criterion:
     - All 17 projects mentioned? ✓
     - Exact names copied? ✓
     - Technologies accurate? ✓
     - Hallucinations? No ✓

Output dict:
  {
    "is_valid": true,
    "verification_feedback": "All 17 projects... Technologies correct..."
  }

Note:
  - Structured output (Pydantic)
  - Does NOT write new answer
  - Only checks the existing answer
  - ONE LLM call
```

#### `check_verification(state: RAGState) -> str`
```python
def check_verification(state: RAGState) -> str:
    """Route based on verification result."""
    
Input state:
  - is_valid: bool
  - retry_count: int
  
Logic:
  if is_valid == True:
    return "end"           # Finish, show answer
  
  elif retry_count < 1:
    return "retry"         # Try again (go to refine_query)
  
  else:
    return "end"           # Max retries, finish anyway

Example:
  Scenario 1: Valid answer
    - is_valid = True
    - return "end"
    - User sees answer (✓ Verified)
  
  Scenario 2: Invalid, can retry
    - is_valid = False
    - retry_count = 0
    - return "retry"
    - Goes to refine_query
  
  Scenario 3: Invalid, retries used up
    - is_valid = False
    - retry_count = 1
    - return "end"
    - User sees answer (⚠ Not Verified)

Note:
  - NOT a node
  - Just conditional routing
  - Ensures max 1 retry (MAX_RETRIES = 1)
```

#### `refine_query(state: RAGState) -> dict`
```python
def refine_query(state: RAGState) -> dict:
    """Write better search query on retry."""
    
Input state:
  - question: Original question
  - search_query: Previous search query
  - verification_feedback: Why it failed
  - retry_count: 0 (about to become 1)
  
Process:
  1. Build refine prompt:
     "A previous search didn't work.
     
      Original question: [...]
      Previous search query: [...]
      Verifier feedback: [...]
      
      Write a BETTER search query that targets
      the missing information."
  
  2. Call Groq LLM:
     Example feedback: "May have missed matching projects"
     Groq outputs: Better query with explicit conditions
     "Find all projects where team size is less than 5..."
  
  3. Keep project IDs from original question:
     If question had "P038", ensure refined query has "P038"
  
  4. Increment retry_count:
     retry_count: 0 → 1

Output dict:
  {
    "search_query": "Better query that targets missing info",
    "retry_count": 1
  }

Note:
  - Only runs if first attempt failed
  - ONE LLM call
  - After this, workflow re-runs semantic + keyword retrieval
  - Then generate + verify again
  - If invalid again, graph stops (retry_count = 1)
```

#### `build_graph() -> CompiledGraph`
```python
def build_graph():
    """Assemble LangGraph workflow."""
    
Process:
  1. Create StateGraph(RAGState)
  2. Add all nodes:
     graph.add_node("analyze_query", analyze_query)
     graph.add_node("semantic_retrieve", semantic_retrieve)
     graph.add_node("keyword_retrieve", keyword_retrieve)
     graph.add_node("merge_documents", merge_documents)
     graph.add_node("generate_answer", generate_answer)
     graph.add_node("verify_answer", verify_answer)
     graph.add_node("refine_query", refine_query)
  
  3. Add edges (connections):
     graph.add_edge(START, "analyze_query")
     
     # Parallel branches
     graph.add_edge("analyze_query", "semantic_retrieve")
     graph.add_edge("analyze_query", "keyword_retrieve")
     
     # Fan-in (wait for both)
     graph.add_edge(["semantic_retrieve", "keyword_retrieve"], "merge_documents")
     
     # Main flow
     graph.add_edge("merge_documents", "generate_answer")
     graph.add_edge("generate_answer", "verify_answer")
     
     # Conditional routing
     graph.add_conditional_edges(
       "verify_answer",
       check_verification,
       {"retry": "refine_query", "end": END}
     )
     
     # Retry loop back
     graph.add_edge("refine_query", "semantic_retrieve")
     graph.add_edge("refine_query", "keyword_retrieve")
  
  4. Compile graph:
     return graph.compile()

Returns:
  CompiledGraph: Ready to invoke with questions
```

---

### UI Functions

#### `load_rag_graph()`
```python
@st.cache_resource(show_spinner="Loading indexes...")
def load_rag_graph():
    """Load indexes once per Streamlit server."""
    
What it does:
  1. @st.cache_resource decorator:
     - Function runs only ONCE per Streamlit session
     - Results cached in memory
     - Subsequent calls return cached result
  
  2. Load FAISS index:
     get_vector_store()
     
  3. Load BM25 retriever:
     get_keyword_retriever()
     
  4. Load project metadata:
     get_project_metadata()
  
  5. Build and compile graph:
     return build_graph()

Returns:
  CompiledGraph: Ready to invoke
  
Benefit:
  - No reloading on every Streamlit rerun
  - Fast response after initial load
```

#### `show_result(result: dict)`
```python
def show_result(result: dict):
    """Display answer, sources, and filters."""
    
Parameters:
  result: {
    "answer": "...",
    "documents": [...],
    "metadata_matches": [...],
    "is_valid": bool,
    "verification_feedback": "...",
    "retry_count": int,
    ...
  }
  
Process:
  1. Show Answer:
     st.subheader("Answer")
     st.markdown(result["answer"])
  
  2. Show Verification Status:
     if is_valid:
       st.caption("✅ Verified: ...")
     else:
       st.warning("⚠ Not verified: ...")
  
  3. Show Sources Table:
     summarize_sources(documents)
     → Extract unique projects
     → Display as dataframe with project ID, type, pages
  
  4. Expandable Sections:
     - "Projects matching metadata filters (N)"
       Shows all N projects that match conditions
     - "Retrieval details"
       Shows search query and retry count
```

#### `main()`
```python
def main():
    """Streamlit app main function."""
    
Process:
  1. Set page config:
     st.set_page_config(page_title="Project Repository RAG")
  
  2. Check if indexes exist:
     If any missing → show error, stop
     
  3. Load graph (cached):
     graph = load_rag_graph()
  
  4. Create UI:
     - Title: "Project Repository RAG"
     - Caption with description
     - Text input for question
     - "Ask" button (primary button)
  
  5. On button click:
     question = st.text_input value
     
     with st.spinner("Retrieving, answering..."):
       try:
         result = graph.invoke(
           {"question": question},
           config={"run_name": "Project Repository RAG"}
         )
       except Exception as e:
         st.error(f"Error: {e}")
         st.stop()
     
     show_result(result)
  
  6. LangSmith tracing:
     config={"run_name": "..."} enables tracing
     All node executions logged to LangSmith
```

---

## Code Walkthrough

### Ingestion Workflow (`python ingestion/build_index.py`)

```python
# Step 1: Load and group pages
pages = load_pdf()
# Result: 201 Document objects (one per page)

projects = group_pages_by_project(pages)
# Result: 100 dictionaries like {"project_id": "P001", "pages": [...]}

# Step 2: Extract metadata
project_records = []
chunks = []

for project in projects:
    try:
        # Extract info from first page only
        metadata = extract_project_metadata(project)
        # Result: {
        #   "project_id": "P001",
        #   "project_name": "...",
        #   "team_size": 4,
        #   "duration": "3 months",
        #   "duration_weeks": 13.0,
        #   "technology_stack": [...],
        #   ...
        # }
        
        # Build full project text
        full_text = build_project_text(project)
        
        # Create chunks from all pages
        project_chunks = split_project_into_chunks(project, metadata)
        # Result: 2-3 chunks per project
        
        # Store record
        project_records.append({
            **metadata,
            "content": full_text
        })
        chunks.extend(project_chunks)
        
    except ValueError as e:
        # Log error, skip project
        logger.warning(f"Skipping {project['project_id']}: {e}")

# Result: 100 project records, 200 chunks

# Step 3: Create embeddings and FAISS
embeddings = get_embedding_model()
# Uses HuggingFace API to embed text → 384-dimensional vectors

build_vector_store(chunks, embeddings)
# Embeds all 200 chunks and builds FAISS index
# Saves to: data/faiss_index/

# Step 4: Build BM25
build_keyword_retriever(chunks)
# Creates BM25 index from all chunks
# Saves to: data/bm25.pkl

# Step 5: Create SQLite
create_database()
# Creates projects table in data/project.db

insert_projects(project_records)
# Inserts 100 rows (one per project)
# Each row has: project_id, name, domain, type, team_size, duration, tech_stack, content

# All done! Indexes are persistent on disk.
```

### Query Workflow (`streamlit run ui/app.py → ask question`)

```python
# STEP 1: User enters question
# question = "Which projects use Python and Snowflake?"

# STEP 2: Graph invocation starts
state = {
    "question": "Which projects use Python and Snowflake?",
    # All other fields initialized as empty/None
}

result = graph.invoke(state)

# STEP 3: analyze_query node
# Input: state (with question)
# ONE LLM call returns a filled QueryAnalysis schema
# invoke_structured() tries strict JSON-schema mode first and falls back to
# plain JSON mode if Groq rejects the reply (HTTP 400 output_parse_failed)
analysis = invoke_structured(QUERY_ANALYSIS_PROMPT, QueryAnalysis, {"question": question})
search_query = keep_project_ids(clean_single_line(analysis.search_query) or question, question)
# Output: "technology stack Python Snowflake"

# Validate the conditions in Python
metadata_filters = build_metadata_filters(
    analysis,
    question,
    get_technology_names()  # All tech names from DB
)
# Output: {"technologies": ["Python", "Snowflake"], "technology_match": "all"}

# State updated:
state.update({
    "search_query": "technology stack Python Snowflake",
    "metadata_filters": {"technologies": ["Python", "Snowflake"], ...},
    "retry_count": 0
})

# STEP 4: semantic_retrieve and keyword_retrieve run IN PARALLEL

# Both retrievers first ask which projects satisfy the conditions:
allowed_ids = get_matching_project_ids(state["metadata_filters"])
# None  → no conditions → search all ~200 chunks (k=5)
# {...} → search ONLY the summary chunks of these projects (up to 8)

# semantic_retrieve:
vector_store = get_vector_store()  # Load FAISS (cached)
semantic_docs = vector_store.similarity_search(
    state["search_query"],
    k=8,
    filter=lambda metadata: is_matching_summary_chunk(metadata, allowed_ids),
    fetch_k=vector_store.index.ntotal,  # rank every chunk before filtering
)
# The query is embedded with the same model, compared with every chunk vector,
# and only summary chunks of matching projects are kept

state["semantic_documents"] = semantic_docs

# keyword_retrieve:
bm25 = get_keyword_retriever()  # Load BM25 (cached)
scores = bm25.vectorizer.get_scores(bm25.preprocess_func(state["search_query"]))
# BM25 tokenizes: ["technology", "stack", "python", "snowflake"] and scores all chunks
keyword_docs = [doc for _, doc in sorted(zip(scores, bm25.docs), key=lambda p: p[0], reverse=True)
                if is_matching_summary_chunk(doc.metadata, allowed_ids)][:8]
# Without conditions it is simply: bm25.invoke(state["search_query"])  (top 5)

state["keyword_documents"] = keyword_docs

# STEP 5: merge_documents
# Combine semantic_docs + keyword_docs
merged = {}
for doc in semantic_docs:
    merged[doc.metadata["chunk_id"]] = doc
    merged[doc.metadata["chunk_id"]].metadata["retrieval_method"] = "semantic"

for doc in keyword_docs:
    chunk_id = doc.metadata["chunk_id"]
    if chunk_id in merged:
        merged[chunk_id].metadata["retrieval_method"] = "both"
    else:
        merged[chunk_id] = doc
        merged[chunk_id].metadata["retrieval_method"] = "keyword"

documents = list(merged.values())
# Sorted so "both" comes first

# Keep the full list of matching projects for the prompt
matching_projects = []
if state["metadata_filters"]:
    projects_db = get_project_metadata()  # Load from SQLite (cached)

    matching_projects = filter_projects_by_metadata(
        state["metadata_filters"],
        projects_db
    )
    # Compares: technologies in ["Python", "Snowflake"], match="all"
    # Result: 17 projects that use BOTH
    # No chunks need to be dropped here: the retrievers already searched
    # only the matching projects' summary chunks

state.update({
    "documents": documents,
    "metadata_matches": matching_projects
})

# STEP 6: generate_answer
# Build context
context = format_context(state)
# Includes:
# - PROJECTS MATCHING METADATA FILTERS (17 projects)
# - RETRIEVED PROJECT TEXT (chunks)

chain = ANSWER_PROMPT | get_llm() | StrOutputParser()   # prompts live in graph/prompts.py
answer = chain.invoke({
    "question": state["question"],
    "context": context,
    "feedback_note": "",  # Empty on first pass
})
# Groq reads prompt and generates answer; StrOutputParser returns plain text

state["answer"] = answer

# STEP 7: verify_answer
result = invoke_structured(VERIFICATION_PROMPT, VerificationResult, {
    "question": state["question"],
    "answer": state["answer"],
    "context": context,
})
# Strict JSON-schema mode first; plain JSON mode fallback if Groq rejects the reply
# Groq returns: {
#   "is_valid": true,
#   "feedback": "All 17 projects matched, names correct..."
# }

state.update({
    "is_valid": result.is_valid,
    "verification_feedback": result.feedback
})

# STEP 8: check_verification (routing)
if state["is_valid"]:
    # Go to END
    # Return state to user
else:
    if state["retry_count"] < 1:
        # Go to refine_query
    else:
        # Go to END (max retries)

# (In our example: is_valid=True, so go to END)

# STEP 9: Return to Streamlit
# state contains:
# {
#   "question": "...",
#   "search_query": "...",
#   "documents": [...],
#   "metadata_matches": [17 projects],
#   "answer": "The following 17 projects...",
#   "is_valid": true,
#   "verification_feedback": "All checks passed.",
#   "retry_count": 0
# }

# STEP 10: Streamlit displays result
st.write(state["answer"])
# Shows: "The following 17 projects use both Python and Snowflake: P008, P010, ..."

st.dataframe(sources)
# Shows table of retrieved projects

with st.expander("Projects matching metadata filters"):
    # Shows all 17 with details
```

---

## Data Flow Explanation

### Data Transformation at Each Stage

```
┌─────────────────────────────────────────────────────────────────┐
│ INGESTION PHASE                                                 │
├─────────────────────────────────────────────────────────────────┤

PDF File (binary)
      ↓
PyPDFLoader
      ↓
201 Document objects (text per page)
      ↓
group_pages_by_project()
      ↓
100 project dictionaries {"project_id": "P001", "pages": [2 Document]}
      ↓
extract_project_metadata() + build_project_text()
      ↓
100 project records {
  "project_id": "P001",
  "project_name": "...",
  "team_size": 4,
  "duration": "3 months",
  "duration_weeks": 13.0,
  "technology_stack": ["Python", "XGBoost", ...],
  "content": "Full project text..."
}
      ↓
split_project_into_chunks()
      ↓
200 Document objects with metadata:
{
  "page_content": "Project P001: ... [chunk text]",
  "metadata": {
    "project_id": "P001",
    "project_name": "...",
    "team_size": 4,
    "technology_stack": [...],
    "page": 3,
    "chunk_id": "P001-p3-c0"
  }
}
      ↓
get_embedding_model() + build_vector_store()
      ↓
200 embeddings [0.12, -0.45, 0.89, ...] stored in FAISS
      ↓
build_keyword_retriever()
      ↓
BM25 index with tokenized documents
      ↓
insert_projects()
      ↓
SQLite database with 100 rows

┌─────────────────────────────────────────────────────────────────┐
│ QUERY PHASE                                                     │
├─────────────────────────────────────────────────────────────────┤

User Question (string)
      ↓
analyze_query()
      ↓
search_query (string) + metadata_filters (dict)
      ↓
FAISS embedding of search_query → 384 numbers
      ↓
semantic_retrieve()
      ↓
5 Document objects (most similar chunks)
      ↓
keyword_retrieve()
      ↓
5 Document objects (keyword matches)
      ↓
merge_documents()
      ↓
Merged documents + metadata_matches (17 projects)
      ↓
generate_answer()
      ↓
answer (string)
      ↓
verify_answer()
      ↓
is_valid (bool) + verification_feedback (string)
      ↓
show to user
```

### State Object Evolution

```
Initial state (from user):
{
  "question": "Which projects use Python and Snowflake?"
}

After analyze_query:
{
  "question": "Which projects use Python and Snowflake?",
  "search_query": "technology stack Python Snowflake",
  "metadata_filters": {"technologies": ["Python", "Snowflake"], "technology_match": "all"},
  "retry_count": 0
}

After semantic_retrieve & keyword_retrieve (parallel):
{
  ...(above)...,
  "semantic_documents": [5 documents from FAISS],
  "keyword_documents": [5 documents from BM25]
}

After merge_documents:
{
  ...(above)...,
  "documents": [merged & filtered documents],
  "metadata_matches": [
    {"project_id": "P008", "project_name": "...", "team_size": 4, ...},
    {"project_id": "P010", "project_name": "...", "team_size": 3, ...},
    ... (17 total)
  ]
}

After generate_answer:
{
  ...(above)...,
  "answer": "The following 17 projects use both Python and Snowflake:\n- P008...\n- P010..."
}

After verify_answer:
{
  ...(above)...,
  "is_valid": true,
  "verification_feedback": "All 17 projects match filters. Names and technologies correct."
}

Final state (after routing):
{
  ...(all above)...,
  [ready to return to user]
}
```

---

## How Functions Connect

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPLETE CALL CHAIN                           │
└──────────────────────────────────────────────────────────────────┘

INGESTION:
──────
build_index.py
  ├─ load_pdf()
  ├─ group_pages_by_project(pages)
  ├─ For each project:
  │   ├─ extract_project_metadata(project)
  │   ├─ build_project_text(project)
  │   ├─ split_project_into_chunks(project, metadata)
  │   │   ├─ build_chunk_header(metadata)
  │   │   └─ RecursiveCharacterTextSplitter.split_text()
  │   └─ Store record
  ├─ get_embedding_model()
  ├─ build_vector_store(chunks, embeddings)
  ├─ build_keyword_retriever(chunks)
  ├─ create_database()
  └─ insert_projects(project_records)

QUERY:
────
Streamlit (ui/app.py)
  ├─ main()
  │   ├─ load_rag_graph()
  │   │   ├─ get_vector_store()
  │   │   ├─ get_keyword_retriever()
  │   │   ├─ get_project_metadata()
  │   │   └─ build_graph()
  │   │       ├─ StateGraph(RAGState)
  │   │       ├─ add_node("analyze_query", analyze_query)
  │   │       ├─ add_node("semantic_retrieve", semantic_retrieve)
  │   │       ├─ add_node("keyword_retrieve", keyword_retrieve)
  │   │       ├─ add_node("merge_documents", merge_documents)
  │   │       ├─ add_node("generate_answer", generate_answer)
  │   │       ├─ add_node("verify_answer", verify_answer)
  │   │       ├─ add_node("refine_query", refine_query)
  │   │       ├─ add_edge(START, "analyze_query")
  │   │       ├─ add_edge("analyze_query", ["semantic_retrieve", "keyword_retrieve"])
  │   │       ├─ add_edge([...], "merge_documents")
  │   │       ├─ add_edge("merge_documents", "generate_answer")
  │   │       ├─ add_edge("generate_answer", "verify_answer")
  │   │       ├─ add_conditional_edges("verify_answer", check_verification, ...)
  │   │       └─ compile()
  │   └─ graph.invoke({"question": ...})
  │       ├─ analyze_query(state)
  │       │   ├─ invoke_structured(QUERY_ANALYSIS_PROMPT, QueryAnalysis, ...)
  │       │   └─ build_metadata_filters(analysis, question, get_technology_names())
  │       ├─ semantic_retrieve(state) [PARALLEL]
  │       │   ├─ get_vector_store()
  │       │   └─ similarity_search(search_query, k=5)  (k=8 + filter when conditions exist)
  │       ├─ keyword_retrieve(state) [PARALLEL]
  │       │   ├─ get_keyword_retriever()
  │       │   └─ invoke(search_query)  (score all + filter, up to 8, when conditions exist)
  │       ├─ merge_documents(state)
  │       │   ├─ De-duplicate by chunk_id, tag semantic / keyword / both
  │       │   └─ filter_projects_by_metadata(filters, get_project_metadata()) → full match list
  │       ├─ generate_answer(state)
  │       │   ├─ format_context(state)
  │       │   └─ (ANSWER_PROMPT | get_llm() | StrOutputParser()).invoke(...)
  │       ├─ verify_answer(state)
  │       │   └─ invoke_structured(VERIFICATION_PROMPT, VerificationResult, ...)
  │       ├─ check_verification(state) [ROUTING]
  │       │   └─ return "end" or "retry"
  │       └─ [Optional] refine_query(state)
  │           └─ (REFINE_QUERY_PROMPT | get_llm() | StrOutputParser()).invoke(...)
  │
  │   (all prompts are defined in graph/prompts.py)
  │
  └─ show_result(result)
      ├─ st.markdown(answer)
      ├─ summarize_sources(documents)
      └─ st.dataframe(sources)

LangSmith Tracing:
─────────────────
All above logged to LangSmith:
  - Node execution times
  - LLM prompts & responses
  - Token usage
  - Retrieval results
  - Complete traces
```

---

## Summary

This Hybrid RAG system follows a clear two-phase architecture:

1. **OFFLINE (Ingestion)**
   - Parse PDF → Group projects → Extract metadata → Split chunks
   - Embed chunks → Build FAISS → Build BM25 → Create SQLite
   - All persisted to disk, never rebuilt

2. **ONLINE (Query)**
   - User asks question
   - LangGraph workflow: analyze → retrieve (parallel) → merge → generate → verify
   - Option: retry (refine query + retrieve + generate + verify again)
   - Return answer with sources to user

**Key Design Principles:**
- Separation of concerns (ingestion vs querying)
- Hybrid retrieval (semantic + keyword)
- Structured metadata filtering (LLM understands the conditions, Python compares the numbers)
- Verification & controlled retry (max 1 retry)
- Caching at all levels (fast response)
- Clear data flow through state object
- Minimal LLM calls (only when needed)
