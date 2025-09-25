
### **Konfiguration über Environment Variables**

OCR for scanned pdf with unstructured library

**Basic Configuration:**
```bash
# API Keys
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://....supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...

# Model Configuration  
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

**Agentic Chunking Parameters:**
```bash
RAG_MAX_TOKENS=500              # Maximale Chunk-Größe
RAG_SOFT_MAX_TOKENS=650         # Soft-Limit für große Chunks
RAG_MIN_TOKENS=120              # Minimum für Chunk-Merging
RAG_OVERLAP_TOKENS=40           # Overlap zwischen Chunks

RAG_ENABLE_AGENTIC_FOR_NOTES=true
RAG_NOTE_AGENTIC_MIN_TOKENS=300 # Schwelle für agentisches Chunking bei Notizen
```

Always-On Agentic Retrieval
I've successfully transformed your RAG solution to use "always-on" agentic retrieval with intelligent guards. Here's what has been implemented:

🔧 Configuration Changes
Removed all sidebar controls - The left column with parameter sliders is completely gone
Hardcoded optimal parameters exactly as you specified:
Max rounds: 2 (expandable to 3 via confidence gate)
Candidates per round: 12
Token budget: 2000
Min similarity: 0.55
Quality over speed: ON
Recency half-life: 90 days
Early stopping: 0.75

🛡️ Intelligent Guards (Fully Automatic)

Guard 1: Temporal-Cue-Switch
Detects keywords like "heute", "aktuell", "neu", "neueste", "recent", "latest"
Automatically reduces recency half-life from 90 → 30 days for time-sensitive queries

Guard 2: Confidence-Gate
If confidence < 0.5 after round 2, automatically triggers round 3
Enhanced with more candidates, lower similarity threshold, and expanded search

Guard 3: Recall-Boost
If < 3 results found, automatically lowers min_similarity to 0.50 and increases candidates to 15
Ensures no query goes unanswered due to overly strict filtering


**Agentic Retrieval Parameters:**
```bash
RAG_ROUNDS=3                    # Maximale Such-Runden
K_PER_ROUND=15                  # Kandidaten pro Runde
MIN_SIM=0.55                    # Minimum Similarity Score
RAG_TOKEN_BUDGET=2000           # Kontext-Token-Budget
RECENCY_HALFLIFE_DAYS=30        # Halbwertszeit für Aktualitäts-Bonus
DOC_TYPE_PREFERENCE=pdf         # Bevorzugter Dokumenttyp
RAG_ENABLE_FILTERS=true         # Erweiterte Filter aktivieren
```

### **Database Schema**

**rag_pages** (Haupttabelle für Chunks):
```sql
- id: UUID (Primary Key)
- content: TEXT (Chunk-Inhalt)
- embedding: vector(1536) (OpenAI Embedding)
- url: TEXT (Dokument-Identifier)
- metadata: JSONB (Erweiterte Metadaten)
- page: INTEGER (Seitenzahl)
- page_heading: TEXT (Seiten-Überschrift)
- section_heading: TEXT (Abschnitts-Überschrift)  
- token_count: INTEGER (Präzise Token-Anzahl)
- confidence: REAL (Quality Score 0.0-1.0)
- chunk_number: INTEGER (Position im Dokument)
- created_at: TIMESTAMP
```

**document_metadata** (Dokument-Übersicht):
```sql
- doc_id: UUID (Primary Key)
- title: TEXT (Dokument-Titel)
- source_url: TEXT (Original-Pfad)
- file_size_bytes: BIGINT
- created_at: TIMESTAMP
- file_modified_at: TIMESTAMP (aus Datei-Metadaten)
```

**chat_history** (Konversations-Historie):
```sql
- id: UUID (Primary Key)  
- question: TEXT (Benutzer-Frage)
- answer: TEXT (Agent-Antwort)
- username: TEXT (Benutzer-ID)
- created_at: TIMESTAMP
```

ROLLEN-Konzept:

Was wurde geändert:
✅ Neue Rollenfunktionen implementiert:
get_user_role() - gibt die eine Rolle des Users zurück
has_permission(feature) - prüft hierarchische Berechtigungen
✅ Alle 15 has_role() Aufrufe ersetzt durch:
get_user_role() == "admin" für Admin-Prüfungen
has_permission("chat") für Chat-Berechtigung
has_permission("upload") für Upload-Berechtigung
has_permission("delete") für Lösch-Berechtigung
✅ UI-Verbesserungen in Benutzerverwaltung:
Neuen User anlegen: 3 Checkboxes → 1 Radio Button Group
Bestehende User bearbeiten: 3 Checkboxes → 1 Radio Button Group
Benutzerfreundliche Beschreibungen der Rollen
✅ Hierarchische Berechtigungen:
Admin: kann alles (alle Features)
Data User: kann Dokumente verwalten + chatten
Chatbot User: kann nur chatten + Chat-Historie

---

Passwörter können Benutzer im User-Menü oben rechts selbst ändern.
E-Mail-Recovery ist in dieser App nicht aktiv.
Nach Rollen-Updates müssen betroffene Benutzer sich neu einloggen.
Der Service-Client wird nur serverseitig verwendet.
Alle RLS-Policies bleiben für normale Datenbankabfragen aktiv.

---

**🚀 Dieses System repräsentiert den aktuellen Stand der RAG-Technologie mit agentischen KI-Methoden für optimale Wissensextraktion und -bereitstellung.**