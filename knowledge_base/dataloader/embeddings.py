from lancedb.embeddings import get_registry
from config import C

bedrock_embedding = (
    get_registry()
    .get("bedrock-text")
    .create(
        name=C.BEDROCK_EMBEDDING_MODEL,
        region=C.AWS_REGION,
    )
)
