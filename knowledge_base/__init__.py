
from .dataloader.loader import DocumentProcessor

def create_knowledge_base(
  documents_s3_bucket=None,
  vector_s3_bucket=None,
  lance_table_name=None,
  chunking_strategy="hybrid",
):
  return DocumentProcessor(
    documents_s3_bucket=documents_s3_bucket,
    vector_s3_bucket=vector_s3_bucket,
    lance_table_name=lance_table_name,
    chunking_strategy=chunking_strategy,
  )

__all__ = ["DocumentProcessor", "create_knowledge_base"]

