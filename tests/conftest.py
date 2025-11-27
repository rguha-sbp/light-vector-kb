import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest
from loguru import logger


@pytest.fixture(scope="session")
def temp_lancedb_dir():
    d = tempfile.mkdtemp(prefix="lancedb_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def set_test_env(monkeypatch, temp_lancedb_dir):
    # Ensure project root is on sys.path for 'knowledge_base' imports
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Force local mode and point LanceDB to temp path
    monkeypatch.setenv("USE_LOCAL_LANCEDB", "true")
    monkeypatch.setenv("LOCAL_LANCEDB_PATH", temp_lancedb_dir)
    monkeypatch.setenv("LANCEDB_URI", temp_lancedb_dir)
    monkeypatch.setenv("LANCEDB_TABLE_NAME", "kb_test")
    # Avoid real AWS
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2")
    # Buckets (unused in local tests)
    monkeypatch.setenv("VECTOR_DATA_BUCKET_NAME", "test-vector-bucket")
    monkeypatch.setenv("DOCUMENTS_BUCKET_NAME", "test-docs-bucket")
    logger.remove()
    logger.add(lambda msg: None)  # silence logs during tests
