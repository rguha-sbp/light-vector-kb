import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from knowledge_base.dataloader.models import TextModel
from knowledge_base.dataloader.embeddings import bedrock_embedding


def test_textmodel_defaults():
    dims = getattr(bedrock_embedding, "ndims", lambda: 8)()
    m = TextModel(text="abc", vector=[0.0] * dims)
    assert m.text == "abc"
    assert isinstance(m.vector, list)
