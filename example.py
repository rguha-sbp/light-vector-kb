#!/usr/bin/env python3
"""
Example usage of the knowledge base module.

This script shows how to use the knowledge base to load documents and search them.
"""

import os
import json
import lancedb
from pathlib import Path
from config import C
from knowledge_base import DocumentProcessor, create_knowledge_base

# Configure loguru if needed (optional)
from loguru import logger
logger.add("knowledge_base.log", rotation="10 MB")

def search_knowledge_base(query: str, limit: int = 10, semantic_weight: float = 0.6, bm25_weight: float = 0.4) -> str:
    """Hybrid search (semantic + BM25) over LanceDB knowledge base.

    Returns a JSON string with top merged results ordered by combined score.
    Each result item: {"text", "source", "doc_id", "score", "mode_scores": {"semantic", "bm25"}}.
    """
    logger.debug(f"search_knowledge_base called with query: {query}, limit: {limit}")
    try:
        db = lancedb.connect(C.LANCEDB_URI)
        table = db.open_table(C.LANCEDB_TABLE_NAME)
    except Exception as e:
        return f"{{\"error\": \"Failed to open LanceDB table: {e}\"}}"

    # Prefer true native hybrid if supported
    try:
        native = table.search(query, query_type="hybrid").limit(limit).to_list()
        logger.debug(f"Native search returned {len(native)} results")
        out = []
        for r in native:
            out.append({
                "text": r.get("text"),
                "doc_id": r.get("doc_id"),
                "source": r.get("original_file_name") or r.get("title") or "unknown",
                "score": float(r.get("_relevance", 1.0)),
                "mode_scores": {
                    "semantic": float(r.get("_distance", 0.0)),
                    "bm25": float(r.get("_bm25", 0.0)) if r.get("_bm25") is not None else None,
                },
            })
        return json.dumps({"query": query, "results": out}, ensure_ascii=False)
    except Exception as e:
        logger.debug(f"Native hybrid search failed: {e}")

def main():
    # Optional: Use a specific AWS profile for local credentials
    # Set this to the name in ~/.aws/credentials and ~/.aws/config
    aws_profile = os.environ.get("AWS_PROFILE") or os.environ.get("AWS_DEFAULT_PROFILE") or None
    if aws_profile:
        try:
            import boto3
            boto3.setup_default_session(region_name=C.AWS_REGION, profile_name=aws_profile)
            print(f"Using AWS profile: {aws_profile} (region={C.AWS_REGION})")
        except Exception as e:
            print(f"Warning: failed to set AWS profile '{aws_profile}': {e}")

    print("Configurations:")
    for k, v in C.as_dict().items():
        print(f"  {k}: {v}")

    # Choose mode based on config (local or S3)
    if C.USE_LOCAL_LANCEDB:
        print("Running in LOCAL mode")

        # Ensure local directories exist
        Path(C.LOCAL_LANCEDB_PATH).resolve().mkdir(parents=True, exist_ok=True)
        Path(C.LOCAL_DOCUMENTS_PATH).resolve().mkdir(parents=True, exist_ok=True)

        # Create knowledge base (URI will be local path via config)
        kb = DocumentProcessor(lance_table_name=C.LANCEDB_TABLE_NAME, chunking_strategy=C.CHUNKING_STRATEGY)

        # Bulk process all supported documents under LOCAL_DOCUMENTS_PATH (optionally within a prefix)
        # Example: process everything under ./local_documents/docs/
        print("Processing all local documents under prefix 'docs/'...")
        total_chunks = kb.process_documents(prefix="", publisher_system="local-system")
        print(f"Processed {total_chunks} total chunks (local)")

    # Get statistics
    stats = kb.get_document_stats()
    print(f"Knowledge base stats: {stats}")

if __name__ == "__main__":
    # Local quick-start defaults: toggle here or set env vars.
    # For local mode:
    #   setx USE_LOCAL_LANCEDB true
    #   setx LOCAL_LANCEDB_PATH .\local_lancedb
    #   setx LOCAL_DOCUMENTS_PATH .\local_documents
    # For S3 mode:
    #   setx USE_LOCAL_LANCEDB false
    #   setx VECTOR_DATA_BUCKET_NAME your-vector-bucket
    #   setx DOCUMENTS_BUCKET_NAME your-docs-bucket
    #   setx LANCEDB_URI s3://your-vector-bucket/lancedb/

    # Ensure AWS region and embedding model are set (used for Bedrock embeddings)
    os.environ.setdefault("AWS_REGION", C.AWS_REGION)
    os.environ.setdefault("BEDROCK_EMBEDDING_MODEL", C.BEDROCK_EMBEDDING_MODEL)
    # Hint: to use local AWS config, set one of these before running:
    #   setx AWS_PROFILE myprofile
    #   setx AWS_DEFAULT_PROFILE myprofile
    # Or pass in the environment for a single run:
    #   $env:AWS_PROFILE = "myprofile"; python .\example_usage.py
    main()
