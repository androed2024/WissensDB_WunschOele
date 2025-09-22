"""
Database setup and connection utilities for Supabase with pgvector.
"""

import os
from typing import Dict, List, Optional, Any
from postgrest import ReturnMethod
from dotenv import load_dotenv
from pathlib import Path
from supabase import create_client

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)


class SupabaseClient:
    """
    Client for interacting with Supabase and pgvector.

    Args:
        supabase_url: URL for Supabase instance. Defaults to SUPABASE_URL env var.
        supabase_key: API key for Supabase. Defaults to SUPABASE_SERVICE_ROLE_KEY env var.
    """

    def __init__(
        self, supabase_url: Optional[str] = None, supabase_key: Optional[str] = None
    ):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase URL and key must be provided either as arguments or environment variables."
            )

        self.client = create_client(self.supabase_url, self.supabase_key)

    def insert_embedding(
        self, text: str, metadata: dict, embedding: list, url: Optional[str] = None
    ):
        row = {
            "content": text,
            "metadata": metadata,
            "embedding": embedding,
        }
        if url:
            row["url"] = url

    def insert_embedding(
        self,
        text: str,
        metadata: dict,
        embedding: list,
        url: Optional[str] = None,
        chunk_number: int = 0,
    ):
        """
        Speichert einen einzelnen Chunk mit Text, Metadaten und Vektor in die Tabelle 'rag_pages'.
        Optional kann eine URL (z. B. Titel + Timestamp) mitgegeben werden.
        """
        row = {
            "content": text,
            "metadata": metadata,
            "embedding": embedding,
            "chunk_number": chunk_number,  # 🧩 Hier setzen!
        }

        if url:
            row["url"] = url

        try:
            response = self.client.table("rag_pages").insert(row).execute()
            return response
        except Exception as e:
            print(f"❌ Fehler beim Einfügen des Embeddings: {e}")
            raise e

    # database/setup.py (oder wo deine SupabaseClient-Klasse definiert ist)
    def store_document_chunk(
        self, url, chunk_number, content, embedding, metadata,
        chunk_id=None, page_heading=None, section_heading=None,
        token_count=None, confidence=None,
    ):
        data = {
            "url": url,
            "chunk_number": chunk_number,
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
        }
        if chunk_id is not None:
            data["chunk_id"] = chunk_id
        if page_heading is not None:
            data["page_heading"] = page_heading
        if section_heading is not None:
            data["section_heading"] = section_heading
        if token_count is not None:
            data["token_count"] = token_count
        if confidence is not None:
            data["confidence"] = confidence

        resp = self.client.table("rag_pages").insert(data).execute()
        # <- WICHTIG: Dict statt APIResponse zurückgeben
        return resp.data[0] if getattr(resp, "data", None) else data



    def search_documents(
        self,
        query_embedding: List[float],
        match_threshold: float = 0.5,
        match_count: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        # Konfigurierbare Mindest-Similarity
        match_threshold = float(os.getenv("RAG_MIN_SIM", "0.55"))

        params = {
            "query_embedding": query_embedding,
            "match_threshold": match_threshold,
            "match_count": match_count,
        }

        if filter_metadata:
            params["filter"] = filter_metadata

        try:
            result = self.client.rpc("match_rag_pages", params).execute()
            if not result.data:
                print("⚠️ Keine Dokument-Treffer für die Anfrage gefunden.")
                return []

            # Zusätzliche lokale Filterung nach Similarity
            filtered_results = [r for r in result.data if r.get("similarity", 0) >= match_threshold]

            print(f"\n🔍 Top {len(filtered_results)} RAG-Matches (von {len(result.data)} gefiltert):")
            for r in filtered_results:
                score = r.get("similarity", 0.0)
                preview = r["content"][:120].replace("\n", " ")
                print(f"  • Score: {score:.3f} → {preview}...")

            return filtered_results

        except Exception as e:
            print("❌ Fehler bei Supabase-RPC:", str(e))
            return []

    def keyword_search_documents(
        self,
        query: str,
        match_count: int = 20,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        try:
            qb = self.client.table("rag_pages").select(
                "id,url,chunk_number,content,metadata"
            )
            if filter_metadata:
                for key, value in filter_metadata.items():
                    qb = qb.contains("metadata", {key: value})
            qb = qb.ilike("content", f"%{query}%").limit(match_count)
            result = qb.execute()
            return result.data or []
        except Exception as e:
            print("❌ Fehler bei Keyword-Suche:", e)
            return []

    def get_document_by_id(self, doc_id: int) -> Dict[str, Any]:
        result = self.client.table("rag_pages").select("*").eq("id", doc_id).execute()
        return result.data[0] if result.data else {}

    def get_all_document_sources(self) -> List[str]:
        result = self.client.table("rag_pages").select("url").execute()
        urls = set(item["url"] for item in result.data if result.data)
        return list(urls)

    def count_documents(self) -> int:
        return len(self.get_all_document_sources())

    def delete_documents_by_filename(self, filename: str) -> int:
        try:
            response = (
                self.client.table("rag_pages").delete().eq("url", filename).execute()
            )
            deleted = len(response.data or [])
            print(f"🧹 {deleted} Datenbankeinträge mit url = {filename} gelöscht.")
            return deleted
        except Exception as e:
            print(f"❌ Fehler beim Löschen von {filename} in rag_pages: {e}")
            return 0

    def save_chat_history(self, user_name: str, question: str, answer: str) -> Dict[str, Any]:
        """
        Speichert eine Frage-Antwort-Interaktion in der chat_history Tabelle.
        
        Args:
            user_name: Name des Benutzers (aktuell "admin")
            question: Die Frage des Benutzers
            answer: Die Antwort des KI Chatbots
            
        Returns:
            Dict mit den gespeicherten Daten oder leer bei Fehler
        """
        try:
            # MEZ Zeitzone für created_at wird automatisch von Supabase gesetzt
            data = {
                "user_name": user_name,
                "question": question,
                "answer": answer
            }
            
            result = self.client.table("chat_history").insert(data).execute()
            print(f"✅ Chat-Historie gespeichert: Benutzer={user_name}, Frage={question[:50]}...")
            return result.data[0] if result.data else {}
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Chat-Historie: {e}")
            return {}


def setup_database_tables() -> None:
    """
    Set up the necessary database tables and functions for the RAG system.
    This should be run once to initialize the database.

    Note: This is typically done through the Supabase MCP server in production.
    """
    # This is a placeholder for the actual implementation
    # In a real application, you would use the Supabase MCP server to run the SQL
    pass
