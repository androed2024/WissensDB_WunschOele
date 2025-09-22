""""""

"""
Document ingestion pipeline for processing documents and generating embeddings.
run:  streamlit run ui/app.py   
"""

import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import uuid
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from math import ceil

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
        # Load chunk parameters from environment variables
        chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator()
        self.max_file_size_mb = 10
        self.supabase_client = supabase_client or SupabaseClient()
        logger.info(f"Initialized DocumentIngestionPipeline with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

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
        on_phase: Optional[Callable[[str, int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._check_file(file_path):
            return []
        print("JETZT DOC PROCESSING......", file_path)
        try:
            processor = get_document_processor(file_path, chunker=self.chunker)
            if not processor:
                logger.error(f"Unsupported file type: {file_path}")
                return []

            # Metadaten EINMAL holen
            try:
                processor_metadata = processor.get_metadata(file_path)
            except Exception:
                processor_metadata = {}

            base_meta = (metadata or {}).copy()
            title_guess = os.path.splitext(os.path.basename(file_path))[0]

            doc_title = base_meta.get("title") or processor_metadata.get("pdf_title") or title_guess

            # Zeitkandidaten (ISO Strings)
            pdf_created  = processor_metadata.get("pdf_creation_date")
            pdf_modified = processor_metadata.get("pdf_mod_date")
            fs_ctime     = processor_metadata.get("fs_ctime")
            fs_mtime     = processor_metadata.get("fs_mtime")

            # Basis doc_row für bestehende Schema-Kompatibilität
            doc_row = {
                "title": doc_title,
                "source_url": base_meta.get("source_url") or base_meta.get("original_filename") or os.path.basename(file_path),
                "doc_type": base_meta.get("doc_type") or processor_metadata.get("content_type") or "application/octet-stream",
            }

            # Versuche erweiterte Felder hinzuzufügen (falls Schema aktualisiert)
            try:
                doc_row.update({
                    "file_created_at": pdf_created or fs_ctime,
                    "file_modified_at": pdf_modified or fs_mtime,
                    "file_hash": base_meta.get("file_hash"),
                })
                # Versuche Upsert mit neuen Spalten
                on_conflict_cols = "file_hash" if doc_row.get("file_hash") else "source_url,file_created_at"
                doc_resp = self.supabase_client.client.table("document_metadata").upsert(doc_row, on_conflict=on_conflict_cols).execute()
                doc_id = doc_resp.data[0]["doc_id"]
                logger.info("Doc row (upsert) → title=%s | created(file)=%s | modified(file)=%s | on_conflict=%s",
                            doc_row["title"], doc_row.get("file_created_at"), doc_row.get("file_modified_at"), on_conflict_cols)
            except Exception as schema_error:
                # Fallback: Nutze alte Insert-Methode ohne neue Spalten
                logger.warning(f"New schema not available, using fallback insert: {schema_error}")
                doc_row_basic = {
                    "title": doc_title,
                    "source_url": base_meta.get("source_url") or base_meta.get("original_filename") or os.path.basename(file_path),
                    "doc_type": base_meta.get("doc_type") or processor_metadata.get("content_type") or "application/octet-stream",
                }
                doc_insert = self.supabase_client.client.table("document_metadata").insert(doc_row_basic).execute()
                doc_id = doc_insert.data[0]["doc_id"]
                logger.info("Doc row (insert fallback) → title=%s", doc_row_basic["title"])

        except Exception as e:
            logger.error(f"Error getting document processor: {str(e)}")
            return []

        # P1: Chunking Phase
        try:
            raw_content = processor.extract_text(file_path)
            if not raw_content:
                logger.warning(
                    f"No content extracted from {os.path.basename(file_path)}"
                )
                return []

            # Handle different file types differently
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                # Statt alles zusammen → pro Seite chunken, damit page-Infos erhalten bleiben
                chunks = []
                for page_idx, page_text in enumerate(raw_content, start=1):
                    page_chunks = self.chunker.chunk_text(page_text)
                    # page-Nummer an jeden Chunk hängen
                    for ch in page_chunks:
                        ch["page"] = page_idx
                    chunks.extend(page_chunks)
                logger.info(f"PDF per-page chunked into {len(chunks)} chunks (keine Seiteninfos verloren)")
            elif file_extension == '.txt':
                # For TXT files: Use processor output directly (already chunked)
                chunks = raw_content
                logger.info(f"TXT file processed into {len(chunks)} chunks")
            else:
                # For other file types: treat as raw content that needs chunking
                if isinstance(raw_content, list) and len(raw_content) > 0 and isinstance(raw_content[0], dict):
                    chunks = raw_content  # Already chunked
                else:
                    # Raw text that needs chunking
                    combined_text = "\n\n".join(raw_content) if isinstance(raw_content, list) else str(raw_content)
                    chunks = self.chunker.chunk_text(combined_text)
                logger.info(f"File processed into {len(chunks)} chunks")

            total_chunks = len(chunks)
            if on_phase:
                on_phase("chunk", 0, total_chunks)
            
            # Simulate chunk processing for progress
            processed_chunks = 0
            for i, chunk in enumerate(chunks):
                # Optional: chunk preprocessing/validation here
                processed_chunks += 1
                if on_phase and (processed_chunks % max(1, total_chunks // 10) == 0 or processed_chunks == total_chunks):
                    on_phase("chunk", processed_chunks, total_chunks)

            logger.info(
                f"Final processing: {len(chunks)} chunks from {os.path.basename(file_path)}"
            )
        except Exception as e:
            logger.error(f"Failed to extract text: {str(e)}")
            return []

        # P2: Embeddings Phase
        try:
            from document_processing.utils import preprocess_text

            # Robust handling of different chunk formats
            chunk_texts = []
            for chunk in chunks:
                if isinstance(chunk, dict):
                    # Standard format: {"text": "...", "page": ...}
                    if "text" in chunk:
                        chunk_texts.append(preprocess_text(chunk["text"]))
                    else:
                        logger.warning(f"Chunk dictionary missing 'text' key: {chunk}")
                        chunk_texts.append(preprocess_text(str(chunk)))
                elif isinstance(chunk, str):
                    # Legacy format: just strings
                    chunk_texts.append(preprocess_text(chunk))
                else:
                    # Unexpected format
                    logger.warning(f"Unexpected chunk type: {type(chunk)}, converting to string")
                    chunk_texts.append(preprocess_text(str(chunk)))
            
            # Get batch size from environment or use default
            embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", "5"))
            
            total_embed_batches = ceil(len(chunk_texts) / embed_batch_size)
            
            if on_phase:
                on_phase("embed", 0, total_embed_batches)
            
            # Create progress callback for embedding batches
            def embed_progress_callback(current_batch, total_batches):
                if on_phase:
                    on_phase("embed", current_batch, total_batches)
            
            # Use embed_batch with progress callback
            embeddings = self.embedding_generator.embed_batch(
                chunk_texts, 
                batch_size=embed_batch_size,
                progress_callback=embed_progress_callback
            )

            if len(embeddings) != len(chunks):
                logger.warning("Mismatch between chunks and embeddings")
                chunks = chunks[: len(embeddings)]
                chunk_texts = chunk_texts[: len(embeddings)]
        except Exception as e:
            import traceback
            logger.error(f"Error generating embeddings: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

        # P3: Database Insert Phase
        try:

            timestamp = datetime.now().isoformat()
            metadata = metadata.copy() if metadata else {}
            metadata.update(
                {
                    "filename": os.path.basename(file_path),
                    "original_filename": metadata.get("original_filename"),
                    "signed_url": metadata.get("signed_url"),
                    "file_path": file_path,
                    "file_extension": os.path.splitext(file_path)[1].lower(),
                    "file_size_bytes": os.path.getsize(file_path),
                    "processed_at": timestamp,
                    "chunk_count": len(chunks),
                }
            )

            if on_phase:
                on_phase("db", 0, len(chunks))

            stored_records = []

            # Confidence-Basis je nach Parser
            processor_name = (processor_metadata.get("processor") or "").lower()
            if "pypdf2" in processor_name:
                base_conf = 0.98
            elif "unstructured-ocr" in processor_name:
                base_conf = 0.85
            else:
                base_conf = 0.95

            # Einmalige Schleife: pro Chunk genau EIN Insert
            for i, (chunk, embedding, chunk_text) in enumerate(zip(chunks, embeddings, chunk_texts)):
                # page bestimmen (TXT hat evtl. keine page-Info)
                page = chunk.get("page") or 1

                # token_count (Proxy) – tiktoken kannst du später ergänzen
                token_count = len(chunk_text.split())

                # confidence aktuell parserbasiert
                confidence = base_conf

                # chunk_id: doc_id + page + laufende Nummer
                chunk_id = f"{doc_id}:p{page}:{i+1}"

                # Chunk-Metadaten aufbauen
                chunk_metadata = metadata.copy()
                chunk_metadata["page"] = page
                if "section_heading" in chunk:
                    chunk_metadata["section_heading"] = chunk["section_heading"]
                if "page_heading" in chunk:
                    chunk_metadata["page_heading"] = chunk["page_heading"]

                # EIN Insert mit allen neuen Spalten
                stored_record = self.supabase_client.store_document_chunk(
                    url=metadata.get("original_filename"),
                    chunk_number=i,
                    content=chunk_text,
                    embedding=embedding,
                    metadata=chunk_metadata,
                    # ⇩ neue Spalten
                    chunk_id=chunk_id,
                    page_heading=chunk.get("page_heading"),
                    section_heading=chunk.get("section_heading"),
                    token_count=token_count,
                    confidence=confidence,
                )
                stored_records.append(stored_record)

                # Progress update alle 10 Chunks oder am Ende
                if on_phase and ((i % 10 == 0) or (i == len(chunks) - 1)):
                    on_phase("db", i + 1, len(chunks))

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

        # Use environment variables for consistent chunking
        chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedding_generator = EmbeddingGenerator()
        supabase = SupabaseClient()

        # --- NEW: document_metadata-Eintrag für manuelle Notizen ---
        # Annahmen: metadata enthält von der App:
        #  - "title": Titel der Notiz
        #  - "quelle": Kategorie ("Wissen", "Beratung", ...)
        #  - optional "original_filename" / "storage_filename" wenn in Storage geschrieben
        from datetime import datetime, timezone

        manual_title = (metadata.get("title") or (url or "Notiz")).strip()
        category = metadata.get("quelle") or "Notiz"

        # Quelle/URL robust bestimmen (nie NULL)
        source_url = (
            metadata.get("storage_filename")
            or metadata.get("original_filename")
            or (url or manual_title)
        )

        doc_row = {
            "title": manual_title,
            "source_url": source_url,
            "doc_type": category,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extra": {
                "is_manual": True,
                "category": category,
            },
        }

        try:
            # Upsert optional: wenn du Duplikate pro (source_url, created_at) verhindern willst,
            # kannst du `on_conflict="source_url,created_at"` verwenden – falls es dafür
            # einen UNIQUE-Index gibt. Ohne UNIQUE bitte einfach .insert().
            doc_ins = supabase.client.table("document_metadata").insert(doc_row).execute()
            # Wenn du die doc_id für chunk_id o.Ä. brauchst:
            # doc_id = doc_ins.data[0]["doc_id"]
        except Exception as e:
            logger.error(f"failed to insert manual note into document_metadata: {e}")
        # --- END NEW ---

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
