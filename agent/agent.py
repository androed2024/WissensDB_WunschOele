"""
Main agent definition for the RAG AI agent.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, TypedDict

from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from supabase import create_client, SupabaseException

from dotenv import load_dotenv
from pathlib import Path

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import (
    KnowledgeBaseSearch,
    KnowledgeBaseSearchParams,
    KnowledgeBaseSearchResult,
)
from agent.prompts import RAG_SYSTEM_PROMPT

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# Supabase-Konfiguration (Service-Role-Key!)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

# Lazy-init placeholder
global_supabase = None


def get_supabase_client():
    global global_supabase
    if global_supabase is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
            raise SupabaseException("SUPABASE_URL und SUPABASE_SERVICE_ROLE_KEY müssen gesetzt sein")
        global_supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    return global_supabase


class AgentDeps(TypedDict, total=False):
    kb_search: KnowledgeBaseSearch


class RAGAgent:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        kb_search: Optional[KnowledgeBaseSearch] = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either as an argument or environment variable."
            )

        self.kb_search = kb_search or KnowledgeBaseSearch(owner_agent=self)
        self.search_tool = Tool(self.kb_search.search)

        self.agent = Agent(
            f"openai:{self.model}",
            system_prompt=RAG_SYSTEM_PROMPT,
            tools=[self.search_tool],
        )

    async def query(
        self, question: str, max_results: int = 5, source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        print("\n=== [Agent.query] ===")
        print("Frage:", question)

        deps = AgentDeps(kb_search=self.kb_search)
        result = await self.agent.run(question, deps=deps)
        response = result.output

        kb_results = []
        for tool_call in getattr(result, "tool_calls", []):
            if tool_call.tool.name == "search":
                kb_results = tool_call.result or []

        print(f"🔍 Treffer aus Tool 'search': {len(kb_results)}")
        for i, res in enumerate(kb_results):
            sim = res.get("similarity", 0.0)
            snippet = res.get("content", "").replace("\n", " ")[:100]
            print(f"[{i+1}] Score: {sim:.3f} | {snippet}...")

        return {"response": response, "kb_results": kb_results}

    async def get_available_sources(self) -> List[str]:
        return await self.kb_search.get_available_sources()


# Funktion aus Klasse herausgelöst
def format_source_reference(metadata: Dict[str, Any], short: bool = False) -> str:
    """
    Erzeugt on-demand eine signierte URL für private Supabase-Dokumente oder Notizen.
    Ignoriert vorhandene (veraltete) signed_url-Einträge aus dem Upload.
    """
    filename = metadata.get("original_filename", "Unbekanntes Dokument")
    page = metadata.get("page") or "?"
    bucket = metadata.get("source_filter", "privatedocs")

    logging.debug(f"Erzeuge Signed URL für Datei: {filename} im Bucket: {bucket}")

    if short:
        return filename

    client = get_supabase_client()
    try:
        # Versuche verschiedene Ansätze für TXT-Dateien
        file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if file_extension == 'txt':
            # Für TXT-Dateien versuche mit transform-Optionen
            try:
                res = client.storage.from_(bucket).create_signed_url(
                    filename, 
                    3600, 
                    {
                        "download": True,
                        "transform": {
                            "format": "text",
                            "quality": 100
                        }
                    }
                )
            except Exception:
                # Fallback: Standard signed URL
                res = client.storage.from_(bucket).create_signed_url(filename, 3600)
        else:
            res = client.storage.from_(bucket).create_signed_url(filename, 3600)
            
        signed = res.get("signedURL")
        if not signed:
            logging.error(f"Keine signed URL in Response: {res}")
            return f"**Quelle:** {filename}, Seite {page} (kein Link verfügbar)"
    except Exception as e:
        logging.error(f"Fehler beim Erstellen der signierten URL: {e}")
        return f"**Quelle:** {filename}, Seite {page} (Fehler beim Link-Aufbau)"

    if bucket == "notes":
        return f"**Quelle:** Notiz: {filename}\n\n[Notiz anzeigen]({signed})"

    # Bestimme Dateityp basierend auf Dateiendung
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if file_extension == 'txt':
        # Für TXT-Dateien keine Seitenangabe und anderer Link-Text
        return f"**Quelle:** {filename}\n\n[Textdatei öffnen]({signed})"
    else:
        # Für PDF-Dateien mit Seitenangabe
        if page:
            signed += f"#page={page}"
        page_info = f"Seite {page}" if page else "ohne Seitenangabe"
        return f"**Quelle:** {filename}, {page_info}\n\n[PDF öffnen]({signed})"


# Singleton-Instanz für einfachen Import
agent = RAGAgent()
