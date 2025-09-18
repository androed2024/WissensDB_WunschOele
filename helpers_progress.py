# helpers_progress.py
from math import ceil
from pathlib import Path

def count_text_chars(file_path: str, mime_hint: str | None = None) -> int:
    """
    Zählt die Zeichen in einer Datei basierend auf dem Dateityp.
    
    Args:
        file_path: Pfad zur Datei
        mime_hint: Optional MIME-Type Hinweis
        
    Returns:
        Anzahl der Zeichen im Text
    """
    p = Path(file_path)
    suf = p.suffix.lower()
    
    try:
        # Text-Dateien direkt lesen
        if suf == ".txt" or (mime_hint and "text/plain" in mime_hint):
            return len(Path(file_path).read_text(encoding="utf-8", errors="ignore"))
        
        # PDF-Dateien verarbeiten
        if suf == ".pdf":
            try:
                # Erst pypdf versuchen
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                return sum(len(page.extract_text() or "") for page in reader.pages)
            except Exception:
                try:
                    # Fallback auf pdfminer
                    from io import StringIO
                    from pdfminer.high_level import extract_text_to_fp
                    out = StringIO()
                    with open(file_path, "rb") as f:
                        extract_text_to_fp(f, out, laparams=None, output_type="text", codec=None)
                    return len(out.getvalue())
                except Exception:
                    # Fallback: Schätzung basierend auf Dateigröße
                    return int(Path(file_path).stat().st_size * 0.6)
        
        # Andere Dateitypen: Schätzung basierend auf Dateigröße
        return int(Path(file_path).stat().st_size * 0.6)
        
    except Exception:
        # Fallback bei allen Fehlern
        return int(Path(file_path).stat().st_size * 0.6)

def estimate_total_chunks(char_count: int, chunk_size: int, chunk_overlap: int) -> int:
    """
    Schätzt die Anzahl der Chunks basierend auf Zeichenanzahl und Chunk-Parametern.
    
    Args:
        char_count: Anzahl der Zeichen im Text
        chunk_size: Größe eines Chunks
        chunk_overlap: Überlappung zwischen Chunks
        
    Returns:
        Geschätzte Anzahl der Chunks
    """
    step = max(1, chunk_size - chunk_overlap)
    
    if char_count <= 0:
        return 1
        
    return max(1, ceil(max(0, char_count - chunk_overlap) / step))
