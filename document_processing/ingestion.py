""""""

"""
Document ingestion pipeline for processing documents and generating embeddings.
run:  streamlit run ui/app.py   
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from document_processing.chunker import TextChunker
from document_processing.embeddings import EmbeddingGenerator
from document_processing.processors import get_document_processor
from database.setup import SupabaseClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    def __init__(self, supabase_client: Optional[SupabaseClient] = None):
        self.chunker = TextChunker(chunk_size=2000, chunk_overlap=400)
        self.embedding_generator = EmbeddingGenerator()
        self.max_file_size_mb = 10
        self.supabase_client = supabase_client or SupabaseClient()
        logger.info("Initialized DocumentIngestionPipeline with default components")

    def _check_file(self, file_path: str) -> bool:
        print("🔎 Checking file:", file_path)
        if not os.path.exists(file_path):
            print("❌ File not found!")
            return False

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📏 File size: {file_size_mb:.2f} MB")
        if file_size_mb > self.max_file_size_mb:
            logger.error(
                f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size"
            )
            return False

        return True

    def process_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        on_progress: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        if not self._check_file(file_path):
            return []
        print("JETZT DOC PROCESSING......", file_path)
        try:
            processor = get_document_processor(file_path, chunker=self.chunker)
            if not processor:
                logger.error(f"Unsupported file type: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error getting document processor: {str(e)}")
            return []

        try:
            chunks = processor.extract_text(file_path)
            if not chunks:
                logger.warning(
                    f"No chunks extracted from {os.path.basename(file_path)}"
                )
                return []

            logger.info(
                f"Extracted {len(chunks)} chunks from {os.path.basename(file_path)}"
            )
        except Exception as e:
            logger.error(f"Failed to extract text: {str(e)}")
            return []

        try:
            from document_processing.utils import preprocess_text

            chunk_texts = [preprocess_text(chunk["text"]) for chunk in chunks]
            embeddings = self.embedding_generator.embed_batch(chunk_texts, batch_size=5)

            if len(embeddings) != len(chunks):
                logger.warning("Mismatch between chunks and embeddings")
                chunks = chunks[: len(embeddings)]
                chunk_texts = chunk_texts[: len(embeddings)]
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []

        try:
            document_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            metadata = metadata.copy() if metadata else {}
            metadata.update(
                {
                    "filename": os.path.basename(file_path),
                    "original_filename": metadata.get("original_filename"),  # ✅ NEW
                    "signed_url": metadata.get("signed_url"),  # ✅ NEW
                    "file_path": file_path,
                    "file_extension": os.path.splitext(file_path)[1].lower(),  # ✅ NEW
                    "file_size_bytes": os.path.getsize(file_path),
                    "processed_at": timestamp,
                    "chunk_count": len(chunks),
                }
            )

            stored_records = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_metadata = metadata.copy()
                page = chunk.get("page")
                if page is None:
                    chunk_metadata["page"] = 1  # fallback for .txt-Dateien
                else:
                    chunk_metadata["page"] = page

                try:
                    stored_record = self.supabase_client.store_document_chunk(
                        url=metadata.get("original_filename"),
                        chunk_number=i,
                        content=chunk_texts[i],
                        embedding=embedding,
                        metadata=chunk_metadata,
                    )
                    stored_records.append(stored_record)
                except Exception as e:
                    logger.error(f"Error storing chunk {i}: {str(e)}")

            logger.info(f"Stored {len(stored_records)} chunks in database")
            return stored_records

        except Exception as e:
            logger.error(f"Error creating document records: {str(e)}")
            return []

    def process_text(
        self, content: str, metadata: dict, url: Optional[str] = None
    ) -> List[dict]:
        """
        Verarbeitet manuellen Text und speichert ihn samt Embeddings in Supabase.
        """
        from document_processing.chunker import TextChunker
        from document_processing.embeddings import EmbeddingGenerator
        from database.setup import SupabaseClient

        chunker = TextChunker()
        embedding_generator = EmbeddingGenerator()
        supabase = SupabaseClient()

        chunks = chunker.chunk_text(content)
        for chunk in chunks:
            chunk["metadata"] = metadata

        vectors = embedding_generator.embed_batch([c["text"] for c in chunks])

        for i, (chunk, embedding) in enumerate(zip(chunks, vectors)):
            supabase.insert_embedding(
                text=chunk["text"],
                metadata=chunk["metadata"],
                embedding=embedding,
                url=url,
                chunk_number=i,  # ✅ Index mitgeben
            )

        return chunks
