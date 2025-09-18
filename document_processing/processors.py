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
    Processor for PDF files with intelligent detection between text-based and scanned PDFs.
    Uses PyPDF2 for fast text extraction when possible, falls back to unstructured OCR for scanned PDFs.
    """

    def _has_extractable_text(self, file_path: str, check_pages: int = 3) -> bool:
        """
        Check if PDF has extractable text by testing the first few pages with PyPDF2.
        
        Args:
            file_path: Path to the PDF file
            check_pages: Number of pages to check (default: 3)
            
        Returns:
            True if PDF has extractable text (>50 characters found), False otherwise
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF has pages
                if len(pdf_reader.pages) == 0:
                    return False
                    
                # Check first few pages for text content
                total_text = ""
                pages_to_check = min(check_pages, len(pdf_reader.pages))
                
                for i in range(pages_to_check):
                    page = pdf_reader.pages[i]
                    page_text = page.extract_text()
                    total_text += page_text
                    
                # Consider PDF as text-based if we find more than 50 characters
                has_text = len(total_text.strip()) > 50
                logger.info(f"PDF text detection: {len(total_text.strip())} characters found in first {pages_to_check} pages")
                return has_text
                
        except Exception as e:
            logger.warning(f"Error checking PDF text content with PyPDF2: {str(e)}")
            return False
    
    def _extract_with_pypdf2(self, file_path: str) -> List[str]:
        """
        Extract text from PDF using PyPDF2 for fast processing of text-based PDFs.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of raw text elements (one per page)
        """
        try:
            text_elements = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_elements.append(page_text.strip())
                        
            logger.info(f"[PyPDF2-fast] Extracted text from {len(text_elements)} pages")
            return text_elements
            
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {str(e)}")
            return []
    
    def _extract_with_unstructured(self, file_path: str) -> List[str]:
        """
        Extract text from PDF using unstructured with OCR for scanned PDFs.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of raw text elements
        """
        try:
            from unstructured.partition.pdf import partition_pdf
            import unicodedata
            
            elements = partition_pdf(
                filename=file_path,
                languages=["deu", "eng"],
                extract_images_in_pdf=True,  # Enable OCR for scanned PDFs
                strategy="hi_res"  # High resolution strategy for better OCR
            )

            if not elements:
                logger.warning(f"[Unstructured-OCR] No elements extracted: {file_path}")
                return []

            text_elements = []
            for el in elements:
                if el.text and el.text.strip():
                    # Normalize text
                    text = (
                        unicodedata.normalize("NFKC", el.text)
                        .replace("\u2011", "-")
                        .replace("\u00a0", " ")
                    )
                    text_elements.append(text)

            logger.info(f"[Unstructured-OCR] Extracted {len(text_elements)} text elements")
            return text_elements
            
        except Exception as e:
            logger.error(f"Error extracting text with unstructured: {str(e)}")
            return []

    def extract_text(self, file_path: str) -> List[str]:
        """
        Extract text from PDF using intelligent method selection.
        Returns raw text elements for further processing by ingestion pipeline.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of raw text elements (not chunked)
        """
        # Determine which extraction method to use
        if self._has_extractable_text(file_path):
            # Use fast PyPDF2 extraction for text-based PDFs
            logger.info(f"Using PyPDF2-fast extraction for text-based PDF: {os.path.basename(file_path)}")
            text_elements = self._extract_with_pypdf2(file_path)
            extraction_method = "PyPDF2-fast"
        else:
            # Use unstructured with OCR for scanned PDFs
            logger.info(f"Using unstructured-OCR extraction for scanned PDF: {os.path.basename(file_path)}")
            text_elements = self._extract_with_unstructured(file_path)
            extraction_method = "unstructured-OCR"
        
        if not text_elements:
            logger.warning(f"No text extracted from PDF: {file_path}")
            return []
            
        logger.info(f"[{extraction_method}] Successfully extracted {len(text_elements)} text elements from {os.path.basename(file_path)}")
        return text_elements

    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        metadata = super().get_metadata(file_path)
        metadata["content_type"] = "application/pdf"
        
        # Determine which processor method was/would be used
        if self._has_extractable_text(file_path):
            metadata["processor"] = "PdfProcessor(PyPDF2-fast)"
        else:
            metadata["processor"] = "PdfProcessor(unstructured-OCR)"
            
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
