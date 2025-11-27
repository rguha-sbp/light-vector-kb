from config import C
import re
from typing import Dict, List

from loguru import logger


class DocumentChunker:
    """Advanced document chunker with multiple sophisticated strategies"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = 200,
        strategy: str = "semantic",
    ):
        self.chunk_size = int(chunk_size or C.MAX_CONTEXT_TOKENS)
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy  # "semantic", "structural", "hybrid", "sentence"

        logger.info(
            f"Initialized DocumentChunker with chunk_size={self.chunk_size}, strategy={self.strategy}"
        )

        # Initialize NLTK data if available
        try:
            import nltk

            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            self.nltk_available = True
        except ImportError:
            logger.warning("NLTK not available. Falling back to basic text splitting.")
            self.nltk_available = False

    def _identify_document_structure(self, text: str) -> Dict[str, List[tuple]]:
        """Identify document structure including headers, sections, lists, tables"""
        structure = {
            "headers": [],
            "sections": [],
            "lists": [],
            "tables": [],
            "numbered_items": [],
            "bullet_points": [],
        }

        lines = text.split("\n")

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Detect headers (various patterns)
            header_patterns = [
                r"^(?:CHAPTER|Chapter|SECTION|Section|PART|Part)\s+[IVXLC\d]+[:\.]?\s*(.*)$",
                r"^\d+\.\s*[A-Z][^.]*$",  # "1. INTRODUCTION"
                r"^[A-Z][A-Z\s]{5,}$",  # ALL CAPS headers
                r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$",  # Title Case
                r"^#{1,6}\s+(.*)$",  # Markdown headers
                r"^(.+)\n=+$",  # Underlined headers
                r"^(.+)\n-+$",  # Underlined headers
            ]

            for pattern in header_patterns:
                if re.match(pattern, line_stripped):
                    structure["headers"].append((i, line_stripped, "header"))
                    break

            # Detect numbered lists
            if re.match(r"^\d+\.\s+(.+)$", line_stripped):
                structure["numbered_items"].append((i, line_stripped, "numbered_list"))

            # Detect bullet points
            elif re.match(r"^[•▪▫◦‣⁃]\s+(.+)$", line_stripped) or re.match(
                r"^[-*+]\s+(.+)$", line_stripped
            ):
                structure["bullet_points"].append((i, line_stripped, "bullet_point"))

            # Detect tables (simple heuristic)
            elif "|" in line_stripped and line_stripped.count("|") >= 2:
                structure["tables"].append((i, line_stripped, "table_row"))

        return structure

    def _semantic_chunking(self, text: str) -> List[str]:
        """Advanced semantic chunking based on topic coherence"""
        if not self.nltk_available:
            return self._structural_chunking(text)

        try:
            from nltk.corpus import stopwords
            from nltk.tokenize import sent_tokenize, word_tokenize

            sentences = sent_tokenize(text)
            if len(sentences) <= 3:
                return [text]

            # Calculate sentence similarity based on word overlap
            def sentence_similarity(sent1: str, sent2: str) -> float:
                try:
                    words1 = set(word_tokenize(sent1.lower()))
                    words2 = set(word_tokenize(sent2.lower()))
                    stop_words = set(stopwords.words("english"))

                    # Remove stopwords and short words
                    words1 = {w for w in words1 if len(w) > 2 and w not in stop_words}
                    words2 = {w for w in words2 if len(w) > 2 and w not in stop_words}

                    if not words1 or not words2:
                        return 0.0

                    intersection = words1.intersection(words2)
                    union = words1.union(words2)

                    return len(intersection) / len(union) if union else 0.0
                except (ValueError, TypeError, AttributeError):
                    return 0.0

            # Group sentences by semantic similarity
            chunks = []
            current_chunk = []
            current_length = 0

            similarity_threshold = 0.2  # Adjust based on your needs

            for i, sentence in enumerate(sentences):
                sentence_len = len(sentence)

                # Check if adding this sentence would exceed chunk size
                if current_length + sentence_len > self.chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_len
                else:
                    # Check semantic similarity with previous sentence
                    if current_chunk and i > 0:
                        similarity = sentence_similarity(sentences[i - 1], sentence)

                        # If similarity is low, consider starting new chunk
                        if (
                            similarity < similarity_threshold
                            and current_length > self.chunk_size * 0.5
                        ):
                            chunks.append(" ".join(current_chunk))
                            current_chunk = [sentence]
                            current_length = sentence_len
                        else:
                            current_chunk.append(sentence)
                            current_length += sentence_len
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_len

            # Add the last chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        except Exception as e:
            logger.warning(
                f"Error in semantic chunking: {e}. Falling back to structural chunking."
            )
            return self._structural_chunking(text)

    def _structural_chunking(self, text: str) -> List[str]:
        """Chunk based on document structure (headers, sections, lists)"""
        structure = self._identify_document_structure(text)
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        current_length = 0

        # Get all structural boundaries sorted by line number
        boundaries = []
        for struct_type, items in structure.items():
            for line_num, content, _item_type in items:
                boundaries.append((line_num, struct_type, content))

        boundaries.sort(key=lambda x: x[0])

        boundary_lines = {b[0] for b in boundaries}

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_length = len(line_stripped)

            # Check if this is a structural boundary
            is_boundary = i in boundary_lines

            # If we hit a boundary and have content, finalize current chunk
            if is_boundary and current_chunk and current_length > self.chunk_size * 0.3:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line_stripped] if line_stripped else []
                current_length = line_length

            # If adding this line would exceed chunk size
            elif current_length + line_length > self.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line_stripped] if line_stripped else []
                current_length = line_length
            else:
                if line_stripped:  # Only add non-empty lines
                    current_chunk.append(line_stripped)
                    current_length += line_length

        # Add the last chunk
        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return [chunk for chunk in chunks if chunk.strip()]

    def _hybrid_chunking(self, text: str) -> List[str]:
        """Combine structural and semantic approaches"""
        # First, do structural chunking to respect document boundaries
        structural_chunks = self._structural_chunking(text)

        # Then apply semantic chunking to large structural chunks
        final_chunks = []

        for chunk in structural_chunks:
            if len(chunk) <= self.chunk_size * 1.5:  # Within reasonable size
                final_chunks.append(chunk)
            else:
                # Apply semantic chunking to oversized chunks
                semantic_sub_chunks = self._semantic_chunking(chunk)
                final_chunks.extend(semantic_sub_chunks)

        return final_chunks

    def _sliding_window_chunking(self, text: str) -> List[str]:
        """Sliding window approach with overlap for context preservation"""
        if not self.nltk_available:
            return self._split_by_paragraphs(text)

        try:
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(text)

            if len(sentences) <= 3:
                return [text]

            chunks = []
            chunk_sentences = []
            chunk_length = 0

            for sentence in sentences:
                sentence_len = len(sentence)

                # If adding this sentence exceeds chunk size
                if chunk_length + sentence_len > self.chunk_size and chunk_sentences:
                    chunks.append(" ".join(chunk_sentences))

                    # Calculate overlap in sentences
                    overlap_chars = 0
                    overlap_sentences = []

                    # Add sentences from the end for overlap
                    for sent in reversed(chunk_sentences):
                        if overlap_chars + len(sent) <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_chars += len(sent)
                        else:
                            break

                    chunk_sentences = overlap_sentences + [sentence]
                    chunk_length = sum(len(s) for s in chunk_sentences)
                else:
                    chunk_sentences.append(sentence)
                    chunk_length += sentence_len

            # Add the last chunk
            if chunk_sentences:
                chunks.append(" ".join(chunk_sentences))

            return chunks

        except Exception as e:
            logger.warning(
                f"Error in sliding window chunking: {e}. Falling back to paragraph chunking."
            )
            return self._split_by_paragraphs(text)

    def chunk_text(self, text: str) -> List[str]:
        """Main chunking method that applies the selected strategy"""
        if not text.strip():
            return []

        # Clean and normalize text
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)  # Remove excessive line breaks
        text = re.sub(r"[ \t]+", " ", text)  # Normalize whitespace

        logger.info(f"Chunking text using {self.strategy} strategy")

        if self.strategy == "semantic":
            chunks = self._semantic_chunking(text)
        elif self.strategy == "structural":
            chunks = self._structural_chunking(text)
        elif self.strategy == "hybrid":
            chunks = self._hybrid_chunking(text)
        elif self.strategy == "sliding_window":
            chunks = self._sliding_window_chunking(text)
        else:  # fallback to sentence-based
            chunks = self._sentence_based_chunking(text)

        # Post-process chunks
        processed_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 50:  # Filter out very small chunks
                processed_chunks.append(chunk)

        logger.info(f"Created {len(processed_chunks)} chunks from text")
        return processed_chunks

    # Keep the old methods for backward compatibility
    def _extract_sections_from_text(self, text: str) -> List[str]:
        """Legacy method - now uses the main chunking strategy"""
        return self.chunk_text(text)

    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs with intelligent grouping"""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # If adding this paragraph would exceed chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _sentence_based_chunking(self, text: str) -> List[str]:
        """Use NLTK for sentence-based intelligent chunking"""
        if not self.nltk_available:
            return self._split_by_paragraphs(text)

        try:
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(text)

            chunks = []
            current_chunk = ""

            for sentence in sentences:
                if (
                    len(current_chunk) + len(sentence) > self.chunk_size
                    and current_chunk
                ):
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks
        except Exception as e:
            logger.warning(
                f"Error in sentence-based chunking: {e}. Falling back to paragraph chunking."
            )
            return self._split_by_paragraphs(text)
