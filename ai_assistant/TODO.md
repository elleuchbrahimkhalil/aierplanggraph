# Endpoint Selection Improvement TODO

## Goal
Replace slow/heuristic endpoint scoring with fast semantic retrieval using embeddings + unified endpoint file.

## Steps (Approved Plan Breakdown)

### 1. Unified Endpoints (Current: Step 1/8)
- Create util to merge endpoints.sample.json + overrides + Swagger -> ai_assistant/data/all_endpoints.json (dedupe id).
- Update _load_endpoints() to load single file.

### 2. Embeddings Setup (Step 2/8)
- New file: ai_assistant/embeddings.py (SentenceTransformer all-MiniLM-L6-v2).
- Generate ai_assistant/data/endpoints_embeddings.faiss + metadata.json.

### 3. Vector Retrieval (Step 3/8)
- Refactor retrieve_candidate_endpoints: Embed question -> FAISS top-5 (filter domain), hybrid score.

### 4. Optimize LLM Router (Step 4/8)
- select_endpoint_and_params: LLM only if max sim <0.8.

### 5. Enhanced Params (Step 5/8)
- Always LLM-extract params post-selection.

### 6. Caching (Step 6/8)
- LRU cache for query_emb -> top_endpoints.

### 7. Airflow DAG Refresh (Step 7/8)
- dags/update_embeddings.py.

### 8. Tests/Eval (Step 8/8)
- Test queries, log accuracy/speed.

## Progress
- [x] Plan approved & TODO created.

Next: Confirm Step 1?
