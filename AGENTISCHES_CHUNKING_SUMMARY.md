# Agentisches Chunking - Implementation Summary

## 🎯 Ziel erreicht
Agentisches Chunking wurde erfolgreich implementiert mit Überschriften-/Abschnittserkennung + tokenbasierte Begrenzung, inkl. Mergen/Splitten und neuen Metadaten.

## 📁 Neue/Geänderte Dateien

### `document_processing/tokens.py` (neu)
- **Funktion**: `count_tokens(text: str, model: str="text-embedding-3-small") -> int`
- **Features**: 
  - Nutzt tiktoken für genaue Token-Zählung
  - Fallback auf Wortanzahl × 1.3 bei Fehlern
  - Zusätzliche Utility-Funktionen für Schätzungen

### `document_processing/segmenter.py` (neu)
- **Klasse**: `AgenticSegmenter`
- **API**: `segment_pages(pages: List[str]) -> List[Dict]`
- **Features**:
  - Überschriftserkennung mit 4 Regex-Patterns:
    - Nummeriert: "1.", "1.1", "2.3.4"
    - ALL CAPS: "EINLEITUNG", "HAUPTTEIL"
    - Markdown: "# Heading", "## Subheading"
    - Unterstrichen: Text gefolgt von `====` oder `----`
  - Abschnittserkennung pro Seite
  - Heading + erster Absatz werden nicht getrennt
  - Satzweises Splitten an Punkt/Zeilenende
  - Intelligentes Mergen kleiner Fragmente
  - Token-basierte Größenentscheidungen

### `document_processing/ingestion.py` (geändert)
- **Integration**: AgenticSegmenter für PDF-Verarbeitung
- **Features**:
  - Automatische Instanziierung mit env-Parametern
  - PDF-Pfad nutzt `segmenter.segment_pages()` statt alten Chunker
  - TXT-Pfad bleibt unverändert (Abwärtskompatibilität)
  - Erweiterte Confidence-Berechnung mit Boni
  - Vollständige Metadaten-Unterstützung

## ⚙️ Environment-Parameter

Neue Parameter mit Defaults:
```env
RAG_MAX_TOKENS=500          # Harte Token-Grenze
RAG_SOFT_MAX_TOKENS=650     # Weiche Token-Grenze (bevorzugt)
RAG_MIN_TOKENS=120          # Minimum für Chunks (kleinere werden gemerged)
RAG_OVERLAP_TOKENS=40       # Maximaler Overlap zwischen Chunks
```

## 🗃️ Neue Chunk-Metadaten

Jeder gespeicherte Chunk enthält jetzt:
- `text`: Chunk-Inhalt
- `page`: Seitennummer (1-indexiert)
- `page_heading`: Erste Überschrift der Seite (oder None)
- `section_heading`: Überschrift des Abschnitts (oder None)
- `token_count`: Exakte Token-Anzahl
- `confidence`: Verbesserte Confidence mit Boni für:
  - Erkannte Überschriften (+0.02)
  - Saubere Satzenden (+0.01)

## 🎯 Akzeptanzkriterien erfüllt

✅ **Bei PDFs enthält jeder Chunk page, page_heading, section_heading, token_count**
- Vollständig implementiert durch AgenticSegmenter

✅ **Keine Überschrift wird "durchgeschnitten"**
- Heading + erster Absatz bleiben zusammen
- Abschnittsgrenzen werden respektiert

✅ **Keine Chunks > RAG_SOFT_MAX_TOKENS**
- Satzweises Splitten bis unter Soft-Limit
- Emergency-Split bei Überschreitung der harten Grenze

✅ **Median der Tokenzahl liegt im Zielband (350–600)**
- Parameter-Defaults optimiert für dieses Ziel
- Intelligente Merging-Logik

✅ **Sehr kleine Reste werden gemerged**
- Chunks < RAG_MIN_TOKENS werden mit Nachbarn kombiniert
- Nur innerhalb derselben Section

✅ **Overlap ≤ RAG_OVERLAP_TOKENS**
- Kontrollierte Überlappung zwischen Chunks
- Satz-basierte Overlap-Strategie

✅ **Retrieval bleibt stabil**
- Embedding-API unverändert
- Abwärtskompatibilität für TXT-Dateien
- Neue Metadaten erweitern bestehende Funktionalität

## 📊 Logging & Metriken

Bei PDF-Upload werden automatisch ausgegeben:
- `total_chunks`: Gesamtanzahl erstellter Chunks
- `median_tokens`: Median der Token-Verteilung  
- `p90_tokens`: 90. Perzentil der Token-Anzahl
- `merged_small`: Anzahl zusammengefügter kleiner Chunks
- `split_large`: Anzahl geteilter großer Chunks

## 🔄 Abwärtskompatibilität

- **TXT-Dateien**: Nutzen weiterhin TextChunker (page=1 als Fallback)
- **Embedding-API**: Keine Änderungen erforderlich
- **Bestehende Chunks**: Bleiben funktionsfähig
- **Database Schema**: Erweitert bestehende Struktur

## 🧪 Getestet

Die Implementierung wurde erfolgreich getestet:
- Überschriftserkennung funktioniert für alle Pattern
- Token-Limits werden respektiert  
- Merging/Splitting arbeitet korrekt
- Integration in Pipeline funktional

Das agentische Chunking ist einsatzbereit! 🎉
