
### **Konfiguration über Environment Variables**

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

---

**🚀 Dieses System repräsentiert den aktuellen Stand der RAG-Technologie mit agentischen KI-Methoden für optimale Wissensextraktion und -bereitstellung.**