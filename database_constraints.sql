-- Supabase SQL für DB-seitige Guardrails gegen Duplikate
-- Einmalig in der Supabase SQL-Konsole ausführen

-- Voraussetzung für gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Spalten sicherstellen
ALTER TABLE public.document_metadata
  ADD COLUMN IF NOT EXISTS file_created_at  timestamptz,
  ADD COLUMN IF NOT EXISTS file_modified_at timestamptz,
  ADD COLUMN IF NOT EXISTS file_hash        text;

-- Eindeutigkeit über Fingerprint (bevorzugt)
CREATE UNIQUE INDEX IF NOT EXISTS ux_document_metadata_file_hash
  ON public.document_metadata (file_hash)
  WHERE file_hash IS NOT NULL;

-- Zusätzlicher Guard: gleiche Quelle + identischer Dokument-Zeitpunkt
CREATE UNIQUE INDEX IF NOT EXISTS ux_document_metadata_src_created
  ON public.document_metadata (source_url, file_created_at)
  WHERE file_created_at IS NOT NULL;

-- Optional: Duplikate bei Chunks verhindern (falls Retries)
CREATE UNIQUE INDEX IF NOT EXISTS ux_rag_pages_chunk_id
  ON public.rag_pages (chunk_id)
  WHERE chunk_id IS NOT NULL;
