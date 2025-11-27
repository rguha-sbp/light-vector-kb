from typing import Type
from lancedb.pydantic import LanceModel, Vector, FixedSizeListMixin
import pyarrow as pa
from .embeddings import bedrock_embedding


class TextModel(LanceModel):
    # Allow pyarrow and other non-standard types during schema generation
    model_config = {
        "arbitrary_types_allowed": True
    }
    text: str = bedrock_embedding.SourceField()
    vector: Vector(bedrock_embedding.ndims()) = bedrock_embedding.VectorField()
    doc_id: str = ""
    title: str = ""
    original_file_name: str = ""
    doc_hash: str = ""
    source_s3_key: str = ""
    publisher_connecting_system: str = ""
    page: str = ""
    section: str = ""
    start_token: str = ""
    end_token: str = ""
    ingested_at: str = ""
    updated_at: str = ""
