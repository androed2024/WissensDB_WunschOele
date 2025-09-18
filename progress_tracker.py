# progress_tracker.py
from math import ceil
from collections import defaultdict

class ProgressTracker:
    """
    Phasen-basierter Fortschritts-Tracker für die Dokumenten-Pipeline.
    Mappt Teilfortschritte verschiedener Phasen in definierte Prozentbereiche.
    """
    
    RANGES = {
        "upload":   (0, 15),
        "chunk":    (15, 45),
        "embed":    (45, 85),
        "db":       (85, 97),
        "finalize": (97, 100),
    }

    def __init__(self, filename: str, update_ui, file_size_bytes: int = 0):
        """
        Initialisiert den ProgressTracker.
        
        Args:
            filename: Name der zu verarbeitenden Datei
            update_ui: Callback-Funktion (filename, percent_str, status_str)
            file_size_bytes: Dateigröße in Bytes für KB-Anzeige
        """
        self.filename = filename
        self.update_ui = update_ui  # fn(filename, percent_str, status_str)
        self.total = defaultdict(lambda: 1)
        self.done = defaultdict(int)
        self.file_size_bytes = file_size_bytes

    def set_total(self, phase: str, total: int):
        """
        Setzt die Gesamtanzahl für eine Phase.
        
        Args:
            phase: Name der Phase
            total: Gesamtanzahl der zu verarbeitenden Elemente
        """
        self.total[phase] = max(1, int(total))

    def tick(self, phase: str, inc: int = 1, status: str | None = None, cap99: bool = True):
        """
        Erhöht den Fortschritt für eine Phase und aktualisiert die UI.
        
        Args:
            phase: Name der Phase
            inc: Anzahl der hinzuzufügenden verarbeiteten Elemente
            status: Optionaler Status-Text (verwendet Default wenn None)
            cap99: Begrenzt Fortschritt auf max. 99% (außer finalize-Phase)
        """
        self.done[phase] += inc
        start, end = self.RANGES[phase]
        frac = min(1.0, self.done[phase] / max(1, self.total[phase]))
        pct = int(start + frac * (end - start))
        
        # Verhindert 100% vor Finalisierung
        if cap99 and phase != "finalize":
            pct = min(pct, 99)
        
        # Thread-safe UI update with error handling
        try:
            # Format progress with KB info if file size is available
            if self.file_size_bytes > 0:
                processed_kb = (pct / 100.0) * (self.file_size_bytes / 1024)
                total_kb = self.file_size_bytes / 1024
                progress_str = f"{pct}% ({processed_kb:.1f}KB von {total_kb:.1f}KB)"
            else:
                progress_str = f"{pct}%"
                
            self.update_ui(self.filename, progress_str, status or self._default_status(phase))
        except Exception as e:
            print(f"⚠️ UI-Update Fehler für {self.filename}: {e}")
            # Continue without failing the entire process

    @staticmethod
    def _default_status(phase: str) -> str:
        """
        Gibt den Standard-Status-Text für eine Phase zurück.
        
        Args:
            phase: Name der Phase
            
        Returns:
            Status-Text für die Phase
        """
        return {
            "upload": "📥 Upload...",
            "chunk": "🧠 Verarbeitung (Chunking)...",
            "embed": "🔢 Embeddings werden erstellt...",
            "db": "🗄️ Speichern in Datenbank...",
            "finalize": "🔎 Finalisierung..."
        }.get(phase, "⏳ Arbeite...")
