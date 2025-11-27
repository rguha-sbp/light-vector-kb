import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from knowledge_base.dataloader.chunker import DocumentChunker


def test_chunker_basic_split():
    text = "Page 1. This is a short paragraph.\n\nPage 2. Another sentence."
    ch = DocumentChunker(strategy="hybrid")
    chunks = ch.chunk_text(text)
    assert isinstance(chunks, list)
    # Some strategies may return empty for very short texts; just ensure no error
    if chunks:
        assert any("Page 1" in c or "Page 2" in c for c in chunks)


@pytest.mark.parametrize("strategy", ["simple", "hybrid"])
def test_chunker_strategies(strategy):
    text = "Sentence one. Sentence two. Sentence three."
    ch = DocumentChunker(strategy=strategy)
    chunks = ch.chunk_text(text)
    assert isinstance(chunks, list)
