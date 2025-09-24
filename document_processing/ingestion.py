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
from document_processing.segmenter import AgenticSegmenter
from document_processing.tokens import count_tokens
from database.setup import SupabaseClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    """
    Clamp a value to the range [0, 1] to ensure database constraints are met.
    
    Args:
        value: Value to clamp
        
    Returns:
        Clamped value between 0.0 and 1.0
    """
    return max(0.0, min(1.0, float(value)))


class DocumentIngestionPipeline:
    def __init__(self, supabase_client: Optional[SupabaseClient] = None, db_client=None):
        # Load chunk parameters from environment variables
        chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.segmenter = AgenticSegmenter()  # Uses environment variables for token limits
        self.embedding_generator = EmbeddingGenerator()
        self.max_file_size_mb = 10
        self.supabase_client = supabase_client or SupabaseClient()
        self.db_client = db_client  # Direct Supabase client (e.g., from get_sb_user())
        
    def _get_db_client(self):
        """Get DB client - User-Client takes priority over Service-Client."""
        return self.db_client or self.supabase_client.client
        logger.info(f"Initialized DocumentIngestionPipeline with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        logger.info(f"AgenticSegmenter configured with max_tokens={self.segmenter.max_tokens}, "
                   f"soft_max={self.segmenter.soft_max_tokens}, min_tokens={self.segmenter.min_tokens}")

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
                # Versuche Upsert mit neuen Spalten - first try without on_conflict
                try:
                    # Try simple insert first
                    db = self._get_db_client()
                    doc_resp = db.table("document_metadata").insert(doc_row).execute()
                    doc_id = doc_resp.data[0]["doc_id"]
                    logger.info("Doc row (insert) → title=%s | created(file)=%s | modified(file)=%s",
                               doc_row["title"], doc_row.get("file_created_at"), doc_row.get("file_modified_at"))
                except Exception as insert_error:
                    # If insert fails, try upsert with appropriate on_conflict
                    try:
                        on_conflict_cols = "file_hash" if doc_row.get("file_hash") else "source_url"
                        db = self._get_db_client()
                        doc_resp = db.table("document_metadata").upsert(doc_row, on_conflict=on_conflict_cols).execute()
                        doc_id = doc_resp.data[0]["doc_id"]
                        logger.info("Doc row (upsert) → title=%s | created(file)=%s | modified(file)=%s | on_conflict=%s",
                                   doc_row["title"], doc_row.get("file_created_at"), doc_row.get("file_modified_at"), on_conflict_cols)
                    except Exception as upsert_error:
                        # Last fallback: simple insert without duplicate handling, but keep all metadata
                        logger.warning(f"Upsert failed ({upsert_error}), trying simple insert without deduplication")
                        # Remove conflicting unique fields but keep all other metadata
                        fallback_row = doc_row.copy()
                        fallback_row.pop("file_hash", None)  # Remove file_hash to avoid conflicts
                        db = self._get_db_client()
                        doc_resp = db.table("document_metadata").insert(fallback_row).execute()
                        doc_id = doc_resp.data[0]["doc_id"]
                        logger.info("Doc row (fallback insert) → title=%s | created(file)=%s | modified(file)=%s", 
                                   fallback_row["title"], fallback_row.get("file_created_at"), fallback_row.get("file_modified_at"))
            except Exception as schema_error:
                # Fallback: Nutze alte Insert-Methode ohne neue Spalten
                logger.warning(f"New schema not available, using fallback insert: {schema_error}")
                doc_row_basic = {
                    "title": doc_title,
                    "source_url": base_meta.get("source_url") or base_meta.get("original_filename") or os.path.basename(file_path),
                    "doc_type": base_meta.get("doc_type") or processor_metadata.get("content_type") or "application/octet-stream",
                }
                db = self._get_db_client()
                doc_insert = db.table("document_metadata").insert(doc_row_basic).execute()
                doc_id = doc_insert.data[0]["doc_id"]
                logger.info("Doc row (insert fallback) → title=%s | NOTE: file_created_at/file_modified_at not saved due to old schema", doc_row_basic["title"])
                logger.info("Missing metadata would be: created(file)=%s | modified(file)=%s", pdf_created or fs_ctime, pdf_modified or fs_mtime)

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
                # Use AgenticSegmenter for intelligent chunking with heading detection
                chunks = self.segmenter.segment_pages(raw_content)
                logger.info(f"PDF processed with AgenticSegmenter into {len(chunks)} chunks with heading detection")
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

                # token_count from chunk metadata if available (AgenticSegmenter provides this),
                # otherwise calculate from text
                if isinstance(chunk, dict) and "token_count" in chunk:
                    token_count = chunk["token_count"]
                else:
                    # Fallback to word count estimation
                    token_count = len(chunk_text.split())

                # confidence calculation with bonuses
                confidence = base_conf
                
                # Bonus if heading was detected (section_heading or page_heading)
                if (isinstance(chunk, dict) and 
                    (chunk.get("section_heading") or chunk.get("page_heading"))):
                    confidence += 0.02  # Small bonus for structured content
                
                # Bonus if chunk ends cleanly at sentence boundary
                if chunk_text.rstrip().endswith(('.', '!', '?', '\n')):
                    confidence += 0.01  # Small bonus for clean sentence endings
                
                # Clamp confidence to valid range [0, 1]
                confidence = _clamp01(confidence)

                # chunk_id: doc_id + page + laufende Nummer
                chunk_id = f"{doc_id}:p{page}:{i+1}"

                # Chunk-Metadaten aufbauen
                chunk_metadata = metadata.copy()
                chunk_metadata["page"] = page
                
                # Extract headings from chunk if available
                section_heading = None
                page_heading = None
                if isinstance(chunk, dict):
                    section_heading = chunk.get("section_heading")
                    page_heading = chunk.get("page_heading")
                    
                    # Add to metadata for compatibility
                    if section_heading:
                        chunk_metadata["section_heading"] = section_heading
                    if page_heading:
                        chunk_metadata["page_heading"] = page_heading

                # EIN Insert mit allen neuen Spalten
                stored_record = self.supabase_client.store_document_chunk(
                    url=metadata.get("original_filename"),
                    chunk_number=i,
                    content=chunk_text,
                    embedding=embedding,
                    metadata=chunk_metadata,
                    # ⇩ neue Spalten
                    chunk_id=chunk_id,
                    page_heading=page_heading,
                    section_heading=section_heading,
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
    ) -> dict:
        """
        Verarbeitet manuellen Text und speichert ihn samt Embeddings in Supabase.
        Implementiert Duplikatsprüfung für manuelle Notizen.
        
        Returns:
            dict: Status-Information mit 'status' key ('success' oder 'duplicate')
        """
        import hashlib
        from document_processing.chunker import TextChunker
        from document_processing.embeddings import EmbeddingGenerator
        from database.setup import SupabaseClient
        from datetime import datetime, timezone

        # Use environment variables for consistent chunking
        chunk_size = int(os.getenv('CHUNK_SIZE', 1000))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 200))
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        embedding_generator = EmbeddingGenerator()
        # Use class DB client logic instead of creating new instance
        db = self._get_db_client()

        # Duplikatsprüfung und document_metadata Insert nur für manuelle Notizen
        doc_id = None
        note_hash = None
        has_schema_support = False
        
        if metadata.get("is_manual"):
            manual_title = (metadata.get("title") or (url or "Notiz")).strip()
            category = (metadata.get("quelle") or "Notiz").lower()
            content_trimmed = content.strip()
            
            # Hash berechnen: SHA-256 über "{title}\n{content}\n{category}"
            hash_input = f"{manual_title}\n{content_trimmed}\n{category}"
            note_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()
            
            # Prüfe zuerst ob die neue Schema-Unterstützung vorhanden ist
            try:
                # Test-Query um zu prüfen ob file_hash Spalte existiert
                test_result = db.table("document_metadata").select("file_hash").limit(1).execute()
                has_schema_support = True
                logger.info("Schema supports file_hash - deduplication enabled")
            except Exception as e:
                has_schema_support = False
                logger.info(f"Schema doesn't support file_hash - deduplication disabled: {e}")
            
            # Duplikatsuche nur wenn Schema-Unterstützung vorhanden
            if has_schema_support:
                try:
                    existing = db.table("document_metadata").select("doc_id").eq(
                        "file_hash", note_hash
                    ).eq("title", manual_title).eq("doc_type", metadata.get("quelle", "Notiz")).execute()
                    
                    if existing.data:
                        logger.info(f"Duplicate manual note found for hash {note_hash[:8]}...")
                        return {"status": "duplicate", "hash": note_hash}
                except Exception as e:
                    logger.warning(f"Error checking for duplicates: {e}")
            
            # document_metadata Eintrag vorbereiten
            now_utc = datetime.now(timezone.utc)
            source_url = (
                metadata.get("storage_filename")
                or metadata.get("original_filename")
                or (url or manual_title)
            )
            
            if has_schema_support:
                # Mit neuen Spalten
                doc_row = {
                    "title": manual_title,
                    "source_url": source_url,
                    "doc_type": metadata.get("quelle", "Notiz"),
                    "created_at": now_utc.isoformat(),
                    "file_created_at": now_utc.isoformat(),
                    "file_modified_at": now_utc.isoformat(),
                    "file_hash": note_hash,
                    "extra": {
                        "is_manual": True,
                        "category": metadata.get("quelle", "Notiz"),
                    },
                }
            else:
                # Fallback ohne neue Spalten
                doc_row = {
                    "title": manual_title,
                    "source_url": source_url,
                    "doc_type": metadata.get("quelle", "Notiz"),
                    "extra": {
                        "is_manual": True,
                        "category": metadata.get("quelle", "Notiz"),
                    },
                }
            
            # Versuche document_metadata Insert
            try:
                doc_ins = db.table("document_metadata").insert(doc_row).execute()
                doc_id = doc_ins.data[0]["doc_id"]
                logger.info(f"Inserted manual note into document_metadata: doc_id={doc_id}, has_schema={has_schema_support}")
            except Exception as e:
                logger.warning(f"Failed to insert manual note into document_metadata: {e}")
                # WICHTIG: Nicht abbrechen - Chunks trotzdem speichern!

        # Chunking und Embedding - IMMER ausführen, auch wenn document_metadata fehlschlägt
        
        # Determine chunking strategy for manual notes
        chunks = []
        chunking_method = "classic"  # Default
        
        if metadata.get("is_manual"):
            # Check if we should use agentic chunking for this note
            
            enable_agentic = os.getenv('RAG_ENABLE_AGENTIC_FOR_NOTES', 'true').lower() == 'true'
            min_tokens_for_agentic = int(os.getenv('RAG_NOTE_AGENTIC_MIN_TOKENS', '300'))
            
            note_tokens = count_tokens(content)
            
            if enable_agentic and note_tokens >= min_tokens_for_agentic:
                # Use agentic segmentation for long notes
                chunking_method = "agentic"
                note_title = metadata.get("title") or url or "Notiz"
                chunks = self.segmenter.segment_text(content, title=note_title)
                
                # Add consistent metadata to all chunks
                for chunk in chunks:
                    chunk["metadata"] = metadata.copy()
                    chunk["metadata"]["is_manual"] = True
                    if note_hash:
                        chunk["metadata"]["note_hash"] = note_hash
                    chunk["metadata"]["category"] = metadata.get("quelle", "Notiz")
                    
                logger.info(f"Manual note: agentic=Yes, tokens={note_tokens}, chunks={len(chunks)}")
            else:
                # Use classic chunking for short notes
                chunking_method = "classic"
                chunks = self.chunker.chunk_text(content)
                
                # Add metadata and set default values for compatibility
                for chunk in chunks:
                    chunk["metadata"] = metadata.copy()
                    chunk["metadata"]["is_manual"] = True
                    if note_hash:
                        chunk["metadata"]["note_hash"] = note_hash
                    chunk["metadata"]["category"] = metadata.get("quelle", "Notiz")
                    
                    # Set default values for classic chunks to match agentic format
                    chunk["page"] = 1
                    chunk["page_heading"] = metadata.get("title") or url or "Notiz"
                    chunk["section_heading"] = None
                    chunk["token_count"] = count_tokens(chunk["text"])
                
                logger.info(f"Manual note: agentic=No (tokens={note_tokens} < {min_tokens_for_agentic}), chunks={len(chunks)}")
        else:
            # Non-manual content - use classic chunking
            chunks = self.chunker.chunk_text(content)
            for chunk in chunks:
                chunk["metadata"] = metadata.copy()
                # Set default values for non-manual chunks
                chunk.setdefault("page", 1)
                chunk.setdefault("page_heading", None)
                chunk.setdefault("section_heading", None)
                chunk.setdefault("token_count", count_tokens(chunk["text"]))
        
        # Fallback: If segmenter returns empty, use classic chunker
        if not chunks and metadata.get("is_manual"):
            logger.warning("Agentic segmenter returned no chunks, falling back to classic chunker")
            chunks = chunker.chunk_text(content)
            for chunk in chunks:
                chunk["metadata"] = metadata.copy()
                chunk["metadata"]["is_manual"] = True
                if note_hash:
                    chunk["metadata"]["note_hash"] = note_hash
                chunk["metadata"]["category"] = metadata.get("quelle", "Notiz")
                # Set compatibility values
                chunk["page"] = 1
                chunk["page_heading"] = metadata.get("title") or url or "Notiz"
                chunk["section_heading"] = None
                chunk["token_count"] = count_tokens(chunk["text"])
            chunking_method = "fallback"

        vectors = embedding_generator.embed_batch([c["text"] for c in chunks])

        # Chunks mit korrekter chunk_id und erweiterten Metadaten speichern
        for i, (chunk, embedding) in enumerate(zip(chunks, vectors)):
            # chunk_id Format: "{doc_id}:p1:{laufende_nummer}" wenn doc_id vorhanden
            if doc_id:
                chunk_id = f"{doc_id}:p1:{i+1}"
            else:
                chunk_id = None
            
            # Calculate confidence for notes
            confidence = 0.95  # Base confidence for manual notes
            
            # Bonus if heading was detected
            if chunk.get("section_heading"):
                confidence += 0.02
            
            # Bonus if chunk ends cleanly at sentence boundary
            if chunk["text"].rstrip().endswith(('.', '!', '?', '\n')):
                confidence += 0.01
            
            # Clamp confidence
            confidence = _clamp01(confidence)
                
            supabase.store_document_chunk(
                url=url,
                chunk_number=i,
                content=chunk["text"],
                embedding=embedding,
                metadata=chunk["metadata"],
                chunk_id=chunk_id,
                page_heading=chunk.get("page_heading"),
                section_heading=chunk.get("section_heading"),
                token_count=chunk.get("token_count"),
                confidence=confidence,
            )

        # Enhanced logging for notes
        if metadata.get("is_manual"):
            token_counts = [chunk.get("token_count", 0) for chunk in chunks]
            if token_counts:
                median_tokens = sorted(token_counts)[len(token_counts) // 2]
                p90_tokens = sorted(token_counts)[int(len(token_counts) * 0.9)]
                split_large = sum(1 for tc in token_counts if tc > 650)  # RAG_SOFT_MAX_TOKENS
                merged_small = 0  # Would need to track from segmenter stats
                
                logger.info(f"Note processing complete: method={chunking_method}, "
                           f"total_tokens={sum(token_counts)}, chunks={len(chunks)}, "
                           f"median_tokens={median_tokens}, p90_tokens={p90_tokens}, "
                           f"split_large={split_large}, merged_small={merged_small}")

        logger.info(f"Successfully processed {len(chunks)} chunks for content: {content[:50]}...")
        return {"status": "success", "doc_id": doc_id, "chunks_count": len(chunks)}
