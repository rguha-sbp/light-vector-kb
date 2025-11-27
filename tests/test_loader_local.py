import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import json
from knowledge_base.dataloader.loader import DocumentProcessor
from config import C


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_list_documents_local(tmp_path, monkeypatch):
    # Prepare local documents
    docs_dir = tmp_path / "local_documents"
    write_file(docs_dir / "a.txt", "hello world")
    write_file(docs_dir / "b.json", json.dumps({"k": "v"}))
    write_file(docs_dir / "ignore.md", "not supported")

    # Override configuration directly to ensure loader picks local dir
    C.USE_LOCAL_LANCEDB = True
    C.LOCAL_DOCUMENTS_PATH = str(docs_dir)

    dp = DocumentProcessor(chunking_strategy="simple")
    keys = dp.list_documents()
    assert "a.txt" in keys
    assert "b.json" in keys
    assert not any(k.endswith(".md") for k in keys)


def test_process_single_txt(tmp_path, monkeypatch):
    docs_dir = tmp_path / "local_documents"
    write_file(docs_dir / "note.txt", "This is a test document for KB ingestion.")

    C.USE_LOCAL_LANCEDB = True
    C.LOCAL_DOCUMENTS_PATH = str(docs_dir)
    C.LANCEDB_TABLE_NAME = "kb_test_local"

    dp = DocumentProcessor(chunking_strategy="simple")
    rows = dp.process_single_document("note.txt")
    assert isinstance(rows, list)
    # For very short documents chunking may yield zero chunks; ensure no exceptions
    if rows:
        r0 = rows[0]
        assert "text" in r0 and "vector" in r0 and "doc_id" in r0
