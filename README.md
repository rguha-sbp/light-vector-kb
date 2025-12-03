# light-vector-kb 🚀

A lightweight, local‑first knowledge base built on LanceDB. It ingests common document formats (PDF, DOC/DOCX, XLSX, JSON, YAML, TXT), chunks text intelligently, generates embeddings via Amazon Bedrock, and creates full‑text + vector indexes for fast hybrid search.

This repo is designed to run entirely in local mode by default, requiring no S3 or cloud resources. You can flip to S3-backed mode later by changing configuration.

**Key Features ✨**
- Ingestion from local folder or S3
- Smart chunking strategies (`simple`, `hybrid`)
- Bedrock embeddings with configurable model and region
- Full‑text (BM25) index + vector index for hybrid search
- Idempotent re-ingest via per-document content hash
- Delete/update handling for modified documents
- Simple API surface via `DocumentProcessor`

---

**Supported Formats 📂**
- Text: `txt`, `md`
- Documents: `pdf`, `doc`, `docx`
- Spreadsheets: `xlsx`
- Structured: `json`, `yaml`

You can drop these files into `local_documents/` (or configure S3) and the processor will detect, parse, chunk, and index them automatically.

**LLM/Embedding Providers 🤖**
- Embeddings: Amazon Bedrock (e.g., `amazon.titan-embed-text-v2:0`) with region control via `AWS_REGION`.
- Chat/UI: Strands agent (LLM selection delegated to your AWS Bedrock setup via credentials/profile). Default behavior is local KB retrieval + agent response.

Notes:
- Additional providers (OpenAI, Azure, etc.) can be integrated later by swapping the embedding backend and agent config; the current repo focuses on AWS Bedrock.

---

**Quick Start 🧪**
- Create a virtual environment and install deps:
```
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv sync --frozen
```
- Prepare local directories:
```
mkdir local_documents
mkdir local_lancedb
```
- Drop a few files into `local_documents/` (e.g., `notes.txt`, `config.yaml`).

Run the example script:
```
uv run python .\example.py
```

The script ingests documents and prints knowledge base stats. It uses local mode by default.

---

**Configuration ⚙️**
All configuration lives in `config.py` under the `Configurations` class (imported as `C`). Values are set from environment variables with sensible defaults for local mode.

- `USE_LOCAL_LANCEDB`: `true` to use local LanceDB (default)
- `LOCAL_LANCEDB_PATH`: path for local LanceDB storage (default `./local_lancedb`)
- `LOCAL_DOCUMENTS_PATH`: path for local documents (default `./local_documents`)
- `LANCEDB_URI`: database URI. In local mode, automatically resolved to `LOCAL_LANCEDB_PATH`. In S3 mode, set to `s3://<VECTOR_DATA_BUCKET_NAME>/lancedb/` or a custom URI.
- `LANCEDB_TABLE_NAME`: table name (default `knowledge-base`)
- `VECTOR_DATA_BUCKET_NAME`, `DOCUMENTS_BUCKET_NAME`: buckets for S3 mode (not required in local mode)
- `AWS_REGION`: region used for Bedrock embeddings (default `eu-west-1`)
- `BEDROCK_EMBEDDING_MODEL`: Bedrock model name (default `amazon.titan-embed-text-v2:0`)

To override via environment (PowerShell):
```
setx USE_LOCAL_LANCEDB true
setx LOCAL_LANCEDB_PATH .\local_lancedb
setx LOCAL_DOCUMENTS_PATH .\local_documents
setx LANCEDB_TABLE_NAME kb_demo
setx AWS_REGION eu-west-1
setx BEDROCK_EMBEDDING_MODEL amazon.titan-embed-text-v2:0
```

Switch to S3 mode:
```
setx USE_LOCAL_LANCEDB false
setx VECTOR_DATA_BUCKET_NAME your-vector-bucket
setx DOCUMENTS_BUCKET_NAME your-docs-bucket
setx LANCEDB_URI s3://your-vector-bucket/lancedb/
```

---

**Programmatic Usage 📦**

Import from `knowledge_base` and use `DocumentProcessor` directly, or via the helper `create_knowledge_base`.

Minimal local ingestion:
```
from knowledge_base import DocumentProcessor
from config import C

# Ensure local paths exist (optional, DocumentProcessor will create local docs dir)
import pathlib
pathlib.Path(C.LOCAL_LANCEDB_PATH).resolve().mkdir(parents=True, exist_ok=True)
pathlib.Path(C.LOCAL_DOCUMENTS_PATH).resolve().mkdir(parents=True, exist_ok=True)

dp = DocumentProcessor(lance_table_name=C.LANCEDB_TABLE_NAME, chunking_strategy=C.CHUNKING_STRATEGY)

# Process all supported documents under LOCAL_DOCUMENTS_PATH
processed = dp.process_documents(prefix="", publisher_system="local")
print(f"Processed {processed} chunks")

# Search using DocumentProcessor.search_knowledge_base (JSON output)
response_json = dp.search_knowledge_base("climate change", limit=10)
print(response_json)
```

Using the factory:
```
from knowledge_base import create_knowledge_base
from config import C
kb = create_knowledge_base(chunking_strategy="hybrid")
kb.process_documents(prefix="")
print(kb.get_document_stats())

# Search via the processor helper (preferred)
print(kb.search_knowledge_base("my query", limit=5))
```

---

**Using `example.py` 🔎**

The `example.py` script demonstrates:
- Connecting to LanceDB from config (`C.LANCEDB_URI`)
- Creating/opening the table
- Ingesting local documents
- Printing knowledge base stats
- Performing hybrid search via `DocumentProcessor.search_knowledge_base()`

Run it:
```
uv run python .\example.py
```

Excerpt of how the search works (from `example.py`):
```
from knowledge_base import DocumentProcessor
from config import C

dp = DocumentProcessor(lance_table_name=C.LANCEDB_TABLE_NAME, chunking_strategy=C.CHUNKING_STRATEGY)
print(dp.search_knowledge_base("your query", limit=10))
```

---

**Chunking and Embeddings 🧩**
- Chunking strategy is selected via `DocumentProcessor(strategy=...)`. The default `hybrid` aims for readable sections; `simple` makes basic splits for small docs.
- Embeddings are generated via Amazon Bedrock (model + region from `config.py`). If embeddings fail, the processor will fall back to zero vectors to keep ingestion robust.
- Full‑text index (`create_fts_index`) and vector index (`create_index`) are created/verified at table initialization for fast search.

---

**Using in Your App 🧭**
You can import the package and run ingestion at startup, but consider:
- Ingestion can be expensive; run it on demand or behind a feature flag.
- Ensure configuration (`C`) is set before import if you rely on environment variable overrides. Use `setx` for persistent env or set per-process before starting Python.
- For AWS usage, set credentials via environment or profiles. Example:
```
$env:AWS_PROFILE = "myprofile"; $env:AWS_REGION = "eu-west-1"
```
- If you package this into another app, call `DocumentProcessor.process_documents()` during an initialization phase, then use LanceDB’s `table.search(..., query_type="hybrid")` at request time.

---

**Testing ✅**
Run tests:
```
uv run pytest -q
```
Tests default to local mode and set temporary paths via `tests/conftest.py`.


---

**Install with uv (including chatbot extras) 📦**
- Prerequisites: Python >= 3.12 and `uv` installed.
- Create venv and install base deps:
```
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv sync --frozen || uv sync
```
- Install chatbot optional extras (Streamlit + Strands):
```
uv sync --all-extras
```
- Run the chatbot UI locally:
```
uv run streamlit run chatbot/app.py
```


AWS credentials for embeddings can be provided via:
- Profiles (recommended):
```
$env:AWS_PROFILE = "myprofile"; $env:AWS_REGION = "eu-west-1"
```
- Direct keys (for temporary sessions):
```
$env:AWS_ACCESS_KEY_ID = "..."
$env:AWS_SECRET_ACCESS_KEY = "..."
$env:AWS_SESSION_TOKEN = "..."  # if using MFA/SSO
$env:AWS_REGION = "eu-west-1"
```

---

**Why this over AWS Vectors? 🆚**
- **Hybrid Search**: Full‑text (BM25) + vector search in one query via LanceDB, improving relevance for both exact terms and semantic intent.
- **Chunking Control**: Multiple strategies (`simple`, `hybrid`) with tunable sizes and overlap; you own the preprocessing, not a black box.
- **Per‑Chunk Metadata**: Store rich metadata (source path, document hash, section headers) alongside embeddings for traceability and better reranking.
- **Local‑First Performance**: No network hop for indexing/search; fast iteration and lower latency for development and small/medium deployments.
- **Cost & Portability**: Avoids managed service costs and vendor lock‑in; data lives in your folders or S3 of your choice.
- **Deterministic Re‑ingest**: Hash‑based idempotency ensures only changed documents are reprocessed, saving time and token and compute.

Comparison snapshot:

| Capability | Light Vector KB (LanceDB) | AWS Vectors (managed) |
| --- | --- | --- |
| Search Mode | Hybrid (BM25 + vectors) | Primarily vector; BM25 requires extra wiring |
| Chunking | Pluggable strategies you control | Service‑side defaults; limited customization |
| Metadata | Arbitrary per‑chunk fields | Vectors + limited attributes |
| Hosting | Local or your S3 | Fully managed by AWS |
| Latency | Local IO, very low | Network call overhead |
| Cost | Infra you own; no per‑query fees | Managed pricing per storage/query |
| Portability | Files + LanceDB; easy export | Tied to AWS service APIs |

Notes:
- You can still use AWS (Bedrock) for embeddings while keeping storage/search local with LanceDB.
- For production scale or multi‑tenant setups, AWS managed vectors can simplify operations; this project optimizes for developer control and local performance.

---

**Security & Secrets 🔐**
This repo contains no hardcoded secrets or credentials. Configuration values are read from environment variables with safe defaults. When switching to S3 or using AWS services:
- Provide credentials via standard AWS mechanisms (profiles or environment) and avoid committing them to the repo.
- Do not hardcode bucket names or URIs that reveal internal infrastructure unless intended.

Quick checks you can run:
```
# Verify current Python env and avoid leaking secrets via code
python -c "import sys; print(sys.executable)"
# Confirm no hardcoded keys exist (optional heuristic)
Select-String -Path **/* -Pattern 'AKIA|SECRET|TOKEN' -ErrorAction SilentlyContinue
```

**License 📄**
This repository is intended for local and internal use. Please consult your organization’s policies for external distribution.
