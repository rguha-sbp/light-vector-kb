from __future__ import annotations
import os
from pathlib import Path


class Configurations:
    # Buckets / storage
    VECTOR_DATA_BUCKET_NAME = os.getenv("VECTOR_DATA_BUCKET_NAME", "my-vector-bucket")
    DOCUMENTS_BUCKET_NAME = os.getenv("DOCUMENTS_BUCKET_NAME", "my-documents-bucket")

    # LanceDB
    LANCEDB_TABLE_NAME = os.getenv("LANCEDB_TABLE_NAME", "knowledge-base")
    # Local override support
    LOCAL_LANCEDB_PATH = os.getenv("LOCAL_LANCEDB_PATH", "./local_lancedb")
    USE_LOCAL_LANCEDB = True
    # Local documents directory (used when local mode enabled)
    LOCAL_DOCUMENTS_PATH = os.getenv("LOCAL_DOCUMENTS_PATH", "./local_documents")
    LANCEDB_URI = os.getenv(
        "LANCEDB_URI",
        (str(Path(LOCAL_LANCEDB_PATH).resolve()) if USE_LOCAL_LANCEDB else f"s3://{VECTOR_DATA_BUCKET_NAME}/lancedb/"),
    )

    # Processing
    CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "hybrid")
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4000"))

    # AWS / Embeddings
    AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "eu-west-1"))
    BEDROCK_EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")

    @classmethod
    def as_dict(cls) -> dict:
        return {
            "VECTOR_DATA_BUCKET_NAME": cls.VECTOR_DATA_BUCKET_NAME,
            "DOCUMENTS_BUCKET_NAME": cls.DOCUMENTS_BUCKET_NAME,
            "LANCEDB_TABLE_NAME": cls.LANCEDB_TABLE_NAME,
            "LANCEDB_URI": cls.LANCEDB_URI,
            "CHUNKING_STRATEGY": cls.CHUNKING_STRATEGY,
            "AWS_REGION": cls.AWS_REGION,
            "BEDROCK_EMBEDDING_MODEL": cls.BEDROCK_EMBEDDING_MODEL,
            "USE_LOCAL_LANCEDB": cls.USE_LOCAL_LANCEDB,
            "LOCAL_LANCEDB_PATH": cls.LOCAL_LANCEDB_PATH,
            "LOCAL_DOCUMENTS_PATH": cls.LOCAL_DOCUMENTS_PATH,
        }

    @classmethod
    def refresh(cls) -> None:
        """Reload values from environment (useful in tests)."""
        for key in cls.as_dict().keys():
            setattr(cls, key, os.getenv(key, getattr(cls, key)))


# Convenient alias
C = Configurations
