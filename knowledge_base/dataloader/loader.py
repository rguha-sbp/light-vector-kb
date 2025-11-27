import hashlib
import os
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import boto3
import lancedb
from loguru import logger

from .chunker import DocumentChunker
from .embeddings import bedrock_embedding
from .models import TextModel
from config import C




class DocumentProcessor:
    """Main class for processing documents from storage (S3 or local) and storing in LanceDB.

    Mode selection:
    - Local mode when `C.USE_LOCAL_LANCEDB` is True (documents read from `C.LOCAL_DOCUMENTS_PATH`).
    - S3 mode otherwise (documents read from `DOCUMENTS_BUCKET_NAME`).
    """

    def __init__(
        self,
        documents_s3_bucket: str = None,
        vector_s3_bucket: str = None,
        lance_table_name: str = None,
        chunking_strategy: str = "hybrid",
    ):
        self.documents_s3_bucket = documents_s3_bucket or C.DOCUMENTS_BUCKET_NAME
        self.vector_s3_bucket = vector_s3_bucket or C.VECTOR_DATA_BUCKET_NAME
        self.lance_table_name = lance_table_name or C.LANCEDB_TABLE_NAME
        self.chunker = DocumentChunker(strategy=chunking_strategy)
        self.local_mode = C.USE_LOCAL_LANCEDB
        self.local_documents_path = Path(C.LOCAL_DOCUMENTS_PATH).resolve()
        if self.local_mode:
            self.local_documents_path.mkdir(parents=True, exist_ok=True)

        # Get AWS region from environment
        aws_region = C.AWS_REGION

        if not self.local_mode:
            # Initialize S3 client only in S3 mode
            self.s3_client = boto3.client("s3", region_name=aws_region)
            logger.info(f"AWS context initialized | aws_region={aws_region}")
        else:
            self.s3_client = None
            logger.info(f"Local mode enabled | documents_path={self.local_documents_path}")

        # Initialize LanceDB with S3 backend for vector storage using environment URI
        lancedb_uri = C.LANCEDB_URI
        if not lancedb_uri:
            raise ValueError("LANCEDB_URI configuration is required")

        self.db = lancedb.connect(lancedb_uri)

        # Create or get table with strict LanceModel schema
        try:
            self.table = self.db.open_table(self.lance_table_name)
            logger.info(f"Opened existing LanceDB table | table_name={self.lance_table_name}")
        except (ValueError, FileNotFoundError, Exception):
            try:
                self.table = self.db.create_table(self.lance_table_name, schema=TextModel)
                logger.info(f"Created new LanceDB table with schema | table_name={self.lance_table_name}")
            except Exception as e:
                logger.exception(f"Failed to create LanceDB table with schema | name={self.lance_table_name} error={e}")
                raise

        # Ensure full-text (BM25) index exists for hybrid search
        try:
            if hasattr(self.table, "create_fts_index"):
                # Prefer replace=True if the index already exists
                try:
                    self.table.create_fts_index("text", replace=True)
                except TypeError:
                    # Older SDKs may not support replace arg; fall back to create and ignore 'already exists'
                    self.table.create_fts_index("text")
                try:
                    self.table.wait_for_index(["text_idx"])  # async build
                except Exception:
                    pass
                logger.info("Created/verified FTS (BM25) index on 'text'")
            else:
                logger.warning("FTS indexing API unavailable; hybrid full-text may not work")
        except Exception as e:
            # If index already exists, treat as success
            if "already exists" in str(e).lower():
                logger.info("FTS index 'text_idx' already exists; skipping creation")
            else:
                logger.warning(f"Failed to create FTS index on 'text' | error={e}")

        # Ensure a performant vector index exists for semantic part of hybrid
        try:
            if hasattr(self.table, "create_index"):
                # Explicit vector column to be robust across versions; replace existing if needed
                try:
                    self.table.create_index(
                        metric="cosine",
                        index_type="IVF_PQ",
                        vector_column_name="vector",
                    )
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        raise
                try:
                    self.table.wait_for_index(["vector_idx"])  # name follows convention
                except Exception:
                    pass
                logger.info("Created/verified vector index (IVF_PQ default) on 'vector'")
            else:
                logger.warning("Vector indexing API unavailable; semantic search may be brute-force")
        except Exception as e:
            logger.warning(f"Failed to create vector index on 'vector' | error={e}")

        logger.info(
            f"DocumentProcessor initialized | documents_bucket={self.documents_s3_bucket} "
            f"vector_bucket={self.vector_s3_bucket} table={self.lance_table_name} "
            f"strategy={chunking_strategy}"
        )

    def list_documents(
        self, prefix: str = "", file_extensions: set = {".pdf", ".docx", ".doc", ".xlsx", ".json", ".yaml", ".yml", ".txt"}
    ) -> List[str]:
        """List all supported documents from S3 or local directory depending on mode."""
        if self.local_mode:
            documents: List[str] = []
            base = self.local_documents_path
            prefix_path = (base / prefix).resolve() if prefix else base
            if not prefix_path.exists():
                logger.warning(f"Local prefix path not found | path={prefix_path}")
                return []
            for root, _dirs, files in os.walk(prefix_path):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in file_extensions:
                        # store relative path from base for consistent key semantics
                        rel = os.path.relpath(Path(root) / f, base)
                        documents.append(rel.replace("\\", "/"))
            logger.info(
                f"Listed local documents | base={base} prefix={prefix} count={len(documents)}"
            )
            return documents
        # S3 mode
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.documents_s3_bucket, Prefix=prefix
            )
            documents = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    file_extension = Path(key).suffix.lower()
                    if file_extension in file_extensions:
                        documents.append(key)
            logger.info(
                f"Listed documents from S3 | bucket={self.documents_s3_bucket} "
                f"document_count={len(documents)} prefix={prefix}"
            )
            return documents
        except Exception as e:
            logger.exception(
                f"Error listing documents from S3 | bucket={self.documents_s3_bucket} "
                f"prefix={prefix} error={e}"
            )
            return []

    def _download_s3(self, s3_key: str, local_temp_path: str) -> bool:
        if self.local_mode:
            return False
        try:
            self.s3_client.download_file(
                self.documents_s3_bucket, s3_key, local_temp_path
            )
            logger.debug(
                f"Downloaded document from S3 | s3_key={s3_key} local_path={local_temp_path}"
            )
            return True
        except Exception as e:
            logger.exception(
                f"Error downloading document from S3 | s3_key={s3_key} "
                f"local_path={local_temp_path} error={e}"
            )
            return False

    def extract_text_from_pdf(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from PDF file"""
        try:
            import PyPDF2

            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                metadata = {
                    "total_pages": len(pdf_reader.pages),
                    "title": (
                        pdf_reader.metadata.get("/Title", "")
                        if pdf_reader.metadata
                        else ""
                    ),
                    "author": (
                        pdf_reader.metadata.get("/Author", "")
                        if pdf_reader.metadata
                        else ""
                    ),
                }

                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    text += f"\n[Page {page_num}]\n{page_text}"

                return text, metadata
        except Exception as e:
            logger.exception(f"Error extracting text from PDF | file_path={file_path} error={e}")
            return "", {}

    def extract_text_from_docx(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from DOCX file"""
        try:
            import docx2txt

            text = docx2txt.process(file_path)
            metadata = {
                "file_type": "docx",
                "extracted_images": True,  # docx2txt can extract images too
            }
            return text, metadata
        except Exception as e:
            logger.exception(f"Error extracting text from DOCX | file_path={file_path} error={e}")
            return "", {}

    def extract_text_from_doc(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from legacy .doc files.

        Strategy:
        1) If 'antiword' binary is available on PATH, use it for robust extraction.
        2) Fallback to docx2txt.process (limited support for .doc).
        3) Gracefully handle errors and return empty text if extraction fails.
        """
        import subprocess
        import shutil

        # Try antiword if available
        antiword_path = shutil.which("antiword")
        if antiword_path:
            try:
                result = subprocess.run(
                    [antiword_path, "-m", "UTF-8.txt", file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                text = result.stdout.decode("utf-8", errors="replace")
                metadata = {"file_type": "doc", "extraction": "antiword"}
                if not text.strip():
                    logger.warning(f"antiword produced empty text | file_path={file_path}")
                return text, metadata
            except subprocess.CalledProcessError as e:
                logger.warning(
                    f"antiword failed | file_path={file_path} returncode={e.returncode} stderr={e.stderr.decode(errors='replace')}"
                )
            except Exception as e:
                logger.warning(f"antiword error | file_path={file_path} error={e}")

        # Fallback: docx2txt (limited support for .doc)
        try:
            import docx2txt
            logger.info(f"Falling back to docx2txt for .doc | file_path={file_path}")
            text = docx2txt.process(file_path)
            metadata = {"file_type": "doc", "extraction": "docx2txt"}
            return text, metadata
        except Exception as e:
            logger.exception(f"Error extracting text from DOC | file_path={file_path} error={e}")
            logger.info("For best .doc support, install 'antiword' or convert to .docx using LibreOffice headless")
            return "", {}

    def extract_text_from_xlsx(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from XLSX file"""
        try:
            import openpyxl

            workbook = openpyxl.load_workbook(file_path, data_only=True)
            text = ""
            metadata = {
                "file_type": "xlsx",
                "sheet_count": len(workbook.sheetnames),
                "sheet_names": workbook.sheetnames,
            }

            for sheet_name in workbook.sheetnames:
                worksheet = workbook[sheet_name]
                text += f"\n[Sheet: {sheet_name}]\n"
                
                # Extract all cell values
                for row in worksheet.iter_rows(values_only=True):
                    row_text = []
                    for cell_value in row:
                        if cell_value is not None:
                            row_text.append(str(cell_value))
                    if row_text:
                        text += " ".join(row_text) + "\n"

            return text, metadata
        except Exception as e:
            logger.exception(
                "Error extracting text from XLSX",
                extra={"file_path": file_path, "error": str(e)},
            )
            return "", {}

    def extract_text_from_json(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from JSON file"""
        try:
            import json

            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            # Convert JSON to readable text format
            text = json.dumps(data, indent=2, ensure_ascii=False)
            
            # Also create a flattened version for better searchability
            def flatten_json(obj, prefix=""):
                items = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_key = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, (dict, list)):
                            items.extend(flatten_json(value, new_key))
                        else:
                            items.append(f"{new_key}: {value}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_key = f"{prefix}[{i}]"
                        if isinstance(item, (dict, list)):
                            items.extend(flatten_json(item, new_key))
                        else:
                            items.append(f"{new_key}: {item}")
                return items
            
            flattened = flatten_json(data)
            searchable_text = "\n".join(flattened)
            
            # Combine formatted JSON and flattened version
            text += "\n\n[Flattened Structure]\n" + searchable_text
            
            metadata = {
                "file_type": "json",
                "structure_depth": self._get_json_depth(data),
                "total_keys": len(flattened)
            }
            
            return text, metadata
        except Exception as e:
            logger.exception(
                "Error extracting text from JSON",
                extra={"file_path": file_path, "error": str(e)},
            )
            return "", {}

    def extract_text_from_yaml(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from YAML file"""
        try:
            import yaml

            with open(file_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
            
            # Convert YAML to readable text format
            text = yaml.dump(data, default_flow_style=False, allow_unicode=True)
            
            # Also create a flattened version similar to JSON
            def flatten_yaml(obj, prefix=""):
                items = []
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        new_key = f"{prefix}.{key}" if prefix else key
                        if isinstance(value, (dict, list)):
                            items.extend(flatten_yaml(value, new_key))
                        else:
                            items.append(f"{new_key}: {value}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        new_key = f"{prefix}[{i}]"
                        if isinstance(item, (dict, list)):
                            items.extend(flatten_yaml(item, new_key))
                        else:
                            items.append(f"{new_key}: {item}")
                return items
            
            flattened = flatten_yaml(data)
            searchable_text = "\n".join(flattened)
            
            # Combine formatted YAML and flattened version
            text += "\n\n[Flattened Structure]\n" + searchable_text
            
            metadata = {
                "file_type": "yaml",
                "structure_depth": self._get_json_depth(data),
                "total_keys": len(flattened)
            }
            
            return text, metadata
        except Exception as e:
            logger.exception(
                "Error extracting text from YAML",
                extra={"file_path": file_path, "error": str(e)},
            )
            return "", {}

    def extract_text_from_txt(self, file_path: str) -> tuple[str, Dict[str, Any]]:
        """Extract text from TXT file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
            text = ""
            encoding_used = "utf-8"
            
            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as file:
                        text = file.read()
                    encoding_used = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if not text:
                # Fallback to binary mode with error handling
                with open(file_path, "rb") as file:
                    raw_data = file.read()
                    text = raw_data.decode('utf-8', errors='replace')
                    encoding_used = "utf-8 (with error replacement)"
            
            # Basic text statistics
            lines = text.splitlines()
            word_count = len(text.split())
            
            metadata = {
                "file_type": "txt",
                "encoding": encoding_used,
                "line_count": len(lines),
                "word_count": word_count,
                "character_count": len(text)
            }
            
            return text, metadata
        except Exception as e:
            logger.exception(
                "Error extracting text from TXT",
                extra={"file_path": file_path, "error": str(e)},
            )
            return "", {}

    def _get_json_depth(self, obj, current_depth=0):
        """Helper method to calculate the depth of a nested JSON/YAML structure"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(value, current_depth + 1) for value in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth

    def process_single_document(
        self, key: str, publisher_system: str = ""
    ) -> List[Dict[str, Any]]:
        """Process a single document from storage and return chunk rows (dicts) with upsert behavior.

        Upsert Strategy (Option A):
        - Compute SHA256 content hash of entire extracted text.
        - Query existing rows for this original_file_name.
        - If existing hash matches, skip re-ingest (idempotent).
        - If hash differs, attempt predicate delete of prior rows (if API supports), then insert new chunks.
        - Each chunk row stores doc_hash for future comparisons.
        """
        file_extension = Path(key).suffix.lower()

        # Create a temporary local file
        temp_file_path = f"/tmp/{uuid.uuid4()}{file_extension}"
        local_source_path: Path | None = None
        if self.local_mode:
            local_source_path = (self.local_documents_path / key).resolve()
            if not local_source_path.exists():
                logger.warning(f"Local document not found | path={local_source_path}")
                return []

        try:
            if self.local_mode:
                source_path = str(local_source_path)
            else:
                if not self._download_s3(key, temp_file_path):
                    return []
                source_path = temp_file_path

            # Extract text based on file type
            if file_extension == ".pdf":
                text, metadata = self.extract_text_from_pdf(source_path)
            elif file_extension == ".docx":
                text, metadata = self.extract_text_from_docx(source_path)
            elif file_extension == ".doc":
                text, metadata = self.extract_text_from_doc(source_path)
            elif file_extension == ".xlsx":
                text, metadata = self.extract_text_from_xlsx(source_path)
            elif file_extension == ".json":
                text, metadata = self.extract_text_from_json(source_path)
            elif file_extension in [".yaml", ".yml"]:
                text, metadata = self.extract_text_from_yaml(source_path)
            elif file_extension == ".txt":
                text, metadata = self.extract_text_from_txt(source_path)
            else:
                logger.error(f"Unsupported file type | file_extension={file_extension} key={key}")
                return []

            if not text.strip():
                logger.warning(f"No text extracted from document | key={key}")
                return []

            # Compute content hash for whole document BEFORE chunking
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            # Look for existing rows for this file
            existing_rows = []
            try:
                # Basic semantic search by filename; may include partial matches, filter strictly
                existing_rows = [
                    r
                    for r in self.table.search(Path(key).name).limit(5000).to_list()
                    if r.get("original_file_name") == Path(key).name
                ]
            except Exception as lookup_err:
                logger.warning(f"Existing row lookup failed | key={key} error={lookup_err}")

            prior_hashes = {
                r.get("doc_hash") for r in existing_rows if r.get("doc_hash")
            }
            if prior_hashes and content_hash in prior_hashes:
                logger.info(f"No content change detected, skipping re-ingest | key={key}")
                return []

            if existing_rows and (content_hash not in prior_hashes):
                # Attempt predicate delete
                try:
                    if hasattr(self.table, "delete"):
                        self.table.delete(
                            where=f"original_file_name == '{Path(key).name}'"
                        )
                        logger.info(
                            f"Deleted stale chunks for updated document | key={key} "
                            f"deleted_chunks={len(existing_rows)}"
                        )
                    else:
                        logger.warning(
                                f"Delete API unavailable, stale chunks remain | key={key} "
                            f"stale_chunks={len(existing_rows)}"
                        )
                except Exception as delete_err:
                            logger.warning(f"Failed to delete prior rows | key={key} error={delete_err}")

            # Chunk the text intelligently using the selected strategy
            chunks = self.chunker.chunk_text(text)

            # Create TextModel instances
            text_models = []
            doc_id = str(uuid.uuid4())

            # Prepare clean chunks and batch embed where possible for efficiency
            cleaned_chunks: List[str] = []
            chunk_meta: List[Dict[str, Any]] = []
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue

                # Extract page information from chunk if available
                page_match = re.search(r"\[Page (\d+)\]", chunk)
                page_num = page_match.group(1) if page_match else str(i + 1)

                # Clean the chunk text
                clean_chunk = re.sub(r"\[Page \d+\]\s*", "", chunk).strip()
                cleaned_chunks.append(clean_chunk)
                chunk_meta.append(
                    {
                        "page": page_num,
                        "section": f"section_{i+1}",
                        "length": len(clean_chunk),
                    }
                )

            # Generate embeddings in batch
            try:
                vectors = bedrock_embedding.generate_embeddings(cleaned_chunks)
                if not isinstance(vectors, list) or not vectors:
                    raise ValueError(
                        "generate_embeddings returned empty or invalid result"
                    )
            except Exception as embed_err:
                logger.error(
                    f"Embedding generation failed, using zero vectors fallback | key={key} error={embed_err}"
                )
                vectors = [[0.0] * bedrock_embedding.ndims() for _ in cleaned_chunks]

            # Build dict rows
            for clean_chunk, meta, vec in zip(cleaned_chunks, chunk_meta, vectors):
                now_iso = datetime.now().isoformat()
                row = {
                    "text": clean_chunk,
                    "vector": vec,
                    "doc_id": doc_id,
                    "title": metadata.get("title", Path(key).stem),
                    "original_file_name": Path(key).name,
                    "doc_hash": content_hash,
                    "source_s3_key": key if not self.local_mode else str(local_source_path),
                    "publisher_connecting_system": publisher_system,
                    "page": meta["page"],
                    "section": meta["section"],
                    "start_token": "0",
                    "end_token": str(meta["length"]),
                    "ingested_at": now_iso,
                    "updated_at": now_iso,
                }
                text_models.append(row)

            logger.info(
                f"Document processed successfully | file_name={Path(key).name} "
                f"chunks_created={len(text_models)} doc_id={doc_id}"
            )
            return text_models

        except Exception as e:
            logger.exception(f"Error processing document | key={key} error={e}")
            return []
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_file_path)
            except (OSError, FileNotFoundError):
                pass

    def process_documents(
        self, prefix: str = "", publisher_system: str = ""
    ) -> int:
        """Process all supported documents from storage (local or S3)."""
        supported_extensions = {".pdf", ".docx", ".doc", ".xlsx", ".json", ".yaml", ".yml", ".txt"}
        processed_count = 0
        document_keys = self.list_documents(prefix, supported_extensions)

        for key in document_keys:
            try:
                logger.info(
                    f"Processing document | mode={'local' if self.local_mode else 's3'} key={key}"
                )
                text_models = self.process_single_document(key, publisher_system)

                if text_models:
                    # Add to LanceDB
                    # Add rows (dicts) so automatic embedding kicks in
                    self.table.add(text_models)
                    processed_count += len(text_models)
                    logger.info(
                        f"Added chunks to database | file_name={Path(key).name} chunks_added={len(text_models)}"
                    )

            except Exception as e:
                logger.exception(f"Failed to process document | key={key} error={e}")

        logger.info(
            f"Document processing complete | mode={'local' if self.local_mode else 's3'} total_chunks_processed={processed_count}"
        )
        return processed_count

    def search_documents(self, query: str, limit: int = 10) -> List[Dict]:
        """Search documents in the vector database"""
        try:
            results = self.table.search(query).limit(limit).to_list()
            return results
        except Exception as e:
            logger.exception(f"Search failed | query={query} error={e}")
            return []
    
    def search_knowledge_base(self, query: str, limit: int = 10, semantic_weight: float = 0.6, bm25_weight: float = 0.4) -> str:
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

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            total_chunks = self.table.count_rows()
            unique_docs = len(self.table.to_pandas()["doc_id"].unique())

            return {
                "total_chunks": total_chunks,
                "unique_documents": unique_docs,
                "table_name": self.lance_table_name,
                "documents_s3_bucket": self.documents_s3_bucket,
                "vector_s3_bucket": self.vector_s3_bucket,
            }
        except Exception as e:
            logger.exception(f"Failed to get stats | error={e}")
            return {}

    def delete_document(self, original_file_name: str) -> int:
        """Delete all chunks for a given original file name."""
        try:
            before = self.table.count_rows()
            if hasattr(self.table, "delete"):
                self.table.delete(where=f"original_file_name == '{original_file_name}'")
            else:
                logger.warning(
                    "Table delete method unavailable",
                    extra={"file_name": original_file_name},
                )
                return 0
            after = self.table.count_rows()
            deleted = max(before - after, 0)
            logger.info(
                "Document chunks deleted",
                extra={"file_name": original_file_name, "chunks_deleted": deleted},
            )
            return deleted
        except Exception as e:
            logger.exception(
                "Failed to delete document",
                extra={"file_name": original_file_name, "error": str(e)},
            )
            return 0
