"""
Document processors for PdfProcessor.extract_textextracting text from various file types.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import PyPDF2

# processors.py
from unstructured.partition.pdf import partition_pdf
import unicodedata
from pathlib import Path
from document_processing.chunker import TextChunker

# Set up logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Base class for document processors.
    """

    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from a document file.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content as a string
        """
        raise NotImplementedError("Subclasses must implement extract_text method")

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a document file.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing document metadata
        """
        # Basic metadata common to all file types
        path = Path(file_path)
        return {
            "filename": path.name,
            "file_extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "created_at": path.stat().st_ctime,
            "modified_at": path.stat().st_mtime,
        }


class TxtProcessor(DocumentProcessor):
    """
    Processor for plain text files with robust error handling.
    """

    def __init__(self, chunker: Optional[TextChunker] = None):
        """Initialize TxtProcessor with optional chunker instance."""
        if chunker:
            self.chunker = chunker
            logger.info(f"TxtProcessor using provided chunker with size={chunker.chunk_size}, overlap={chunker.chunk_overlap}")
        else:
            # Fallback: create own chunker with environment variables or defaults
            chunk_size = int(os.getenv('CHUNK_SIZE', 2000))
            chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 400))
            self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            logger.info(f"TxtProcessor created own chunker with size={chunk_size}, overlap={chunk_overlap}")

    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from a TXT file, chunk it, and return list of chunk dictionaries.

        Args:
            file_path: Path to the TXT file

        Returns:
            List of chunk dictionaries with structure {"text": ..., "page": ...}
        """
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try different encodings if UTF-8 fails
        encodings = ["utf-8", "latin-1", "cp1252", "ascii"]
        content = ""

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    content = file.read()
                logger.info(f"Successfully read text file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.warning(
                    f"Failed to decode with {encoding}, trying next encoding"
                )
            except Exception as e:
                logger.error(f"Error reading file with {encoding}: {str(e)}")
                raise

        if not content:
            raise ValueError(
                f"Could not decode file with any of the attempted encodings"
            )

        # Chunk the text using TextChunker
        chunks = self.chunker.chunk_text(content)
        
        logger.info(f"Extracted and chunked text into {len(chunks)} chunks from {path.name}")
        return chunks

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a TXT file.

        Args:
            file_path: Path to the TXT file

        Returns:
            Dictionary containing document metadata
        """
        metadata = super().get_metadata(file_path)
        metadata["content_type"] = "text/plain"
        metadata["processor"] = "TxtProcessor"

        # Count lines and words from chunks
        try:
            chunks = self.extract_text(file_path)
            full_text = " ".join([chunk["text"] for chunk in chunks])
            metadata["line_count"] = len(full_text.splitlines())
            metadata["word_count"] = len(full_text.split())
            metadata["chunk_count"] = len(chunks)
        except Exception:
            # Don't fail metadata collection if text extraction fails
            pass

        return metadata


class PdfProcessor(DocumentProcessor):
    """
    Processor for PDF files using unstructured for better text extraction.
    """

    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        from unstructured.partition.pdf import partition_pdf
        import unicodedata

        elements = partition_pdf(
            filename=file_path,
            languages=["deu", "eng"],
            extract_images_in_pdf=False,
        )

        if not elements:
            logger.warning(f"[Unstructured] No elements extracted: {file_path}")
            return []

        chunks = []

        for el in elements:
            if not el.text:
                continue

            # ✅ Use attribute access for modern unstructured versions
            page = getattr(el.metadata, "page_number", 1)

            text = (
                unicodedata.normalize("NFKC", el.text)
                .replace("\u2011", "-")
                .replace("\u00a0", " ")
            )

            chunks.append({"text": text, "page": page})

        logger.info(
            f"[Unstructured] Extracted {len(chunks)} chunks with pages from {file_path}"
        )
        return chunks

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = super().get_metadata(file_path)
        metadata["content_type"] = "application/pdf"
        metadata["processor"] = "PdfProcessor(unstructured)"
        return metadata


def get_document_processor(file_path: str, chunker: Optional[TextChunker] = None) -> Optional[DocumentProcessor]:
    """
    Get the appropriate processor for a file based on its extension.

    Args:
        file_path: Path to the document file
        chunker: Optional TextChunker instance to use for TXT files

    Returns:
        DocumentProcessor instance for the file type or None if unsupported
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    # Create processor instances
    if extension == ".txt":
        processor = TxtProcessor(chunker=chunker)
    elif extension == ".pdf":
        processor = PdfProcessor()
    else:
        processor = None

    if processor:
        logger.info(f"Using {processor.__class__.__name__} for {path.name}")
        return processor
    else:
        logger.warning(f"Unsupported file type: {extension}")
        return None


def get_supported_extensions() -> List[str]:
    """
    Get a list of supported file extensions.

    Returns:
        List of supported file extensions
    """
    return [".txt", ".pdf"]
