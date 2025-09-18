"""
Streamlit application for the RAG AI agent - WORKING VERSION
Wiederhergestellte Version mit neuer Menüstruktur und Chat-Historie
"""

# Add parent directory to path to allow relative imports
import sys
import os

# Projektbasisverzeichnis zum Pfad hinzufügen (eine Ebene über 'ui')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Logging
import logging

# Date+Time for post knowledge in db
from datetime import datetime
import pytz

from base64 import b64encode

# Reduce verbosity by logging only informational messages and above
logging.basicConfig(level=logging.INFO)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import format_source_reference

import asyncio
import time
from typing import List, Dict, Any
from pathlib import Path
import tempfile
from datetime import datetime

import streamlit as st

import unicodedata
import re

import hashlib


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


from collections import defaultdict

# Logo-Pfad im Root-Verzeichnis
logo_path = "logo-wunschoele.png"

# Logo-Datei als base64 laden
with open(logo_path, "rb") as image_file:
    encoded = b64encode(image_file.read()).decode()


def sanitize_filename(filename: str) -> str:
    filename = filename.strip()
    filename = filename.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    filename = filename.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
    filename = filename.replace("ß", "ss")
    filename = (
        unicodedata.normalize("NFKD", filename)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)
    return filename


from dotenv import load_dotenv

load_dotenv()

import mimetypes

# Import progress tracking modules
from progress_tracker import ProgressTracker
from helpers_progress import count_text_chars, estimate_total_chunks

# Environment variables with defaults
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "5"))
DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "100"))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]


app_version = os.getenv("APP_VERSION", "0.0")
print("DEBUG VERSION:", os.getenv("APP_VERSION"))

from document_processing.ingestion import DocumentIngestionPipeline
from database.setup import SupabaseClient
from agent.agent import RAGAgent, agent as rag_agent, format_source_reference, get_supabase_client
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)

st.set_page_config(
    page_title="Wissens-Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def render_header():
    """Rendert den Header mit aktuellen Dokumenten- und Notizenzählern"""
    doc_count = st.session_state.get("document_count", 0)
    note_count = st.session_state.get("knowledge_count", 0)
    
    st.markdown(
        f"""
        <style>
            @media (max-width: 768px) {{
                .header-flex {{ flex-direction: column; align-items: flex-start; gap: 0.4rem; }}
                .header-title-wrap {{ flex-direction: column; align-items: flex-start; }}
            }}
        </style>
        <div class=\"header-flex\" style=\"display: flex; justify-content: space-between; align-items: center; padding-top: 0.5rem; padding-bottom: 0.5rem;\">
            <div class=\"header-title-wrap\" style=\"display: flex; align-items: center;\">
                <img src=\"data:image/png;base64,{encoded}\" alt=\"Logo\" style=\"height: 42px; margin-right: 14px;\">
                <span style=\"font-size: 22px; font-weight: 600;\">Wunsch-Öle Wissens Agent</span>
                <span style=\"color: #007BFF; font-size: 14px; margin-left: 12px;\">🔧 Version: {app_version}</span>
            </div>
            <div style=\"font-size: 14px;\">
                📄 Dokumente: {doc_count} &nbsp;&nbsp;&nbsp; 🧠 Notizen: {note_count}
            </div>
        </div>
        <hr style=\"margin-top: 0.4rem; margin-bottom: 0.8rem;\">
        """,
        unsafe_allow_html=True,
    )

supabase_client = SupabaseClient()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


# Platzhalter für die fehlenden Funktionen - diese werden später implementiert
async def run_agent_with_streaming(user_input: str):
    # Vereinfachte Version - in der echten App ist das komplexer
    yield "Dies ist eine Platzhalter-Antwort für: " + user_input

async def update_available_sources():
    # Vereinfachte Version
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "document_count" not in st.session_state:
        st.session_state.document_count = 0
    if "knowledge_count" not in st.session_state:
        st.session_state.knowledge_count = 0


async def render_chat_interface():
    """Rendert das Chat-Interface"""
    st.markdown("""
    <style>
    /* Basis-Styling ohne feste Positionierung */
    .stChatInput textarea {
        min-height: 60px !important;
        border-radius: 8px !important;
        border: 2px solid #dee2e6 !important;
        padding: 12px 16px !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #007BFF !important;
        box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
    }
    
    .chat-section {
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header mit gleicher Schriftgröße wie andere Menüs
    st.markdown("<h4>💬 Spreche mit dem Wunsch-Öle KI Assistenten</h4>", unsafe_allow_html=True)

    # Input-Feld ohne Box
    user_input = st.chat_input("Stelle eine Frage zu den Dokumenten...")

    # Chat-Input Verarbeitung
    if user_input:
        st.markdown('<div class="chat-section">', unsafe_allow_html=True)
        
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            async for chunk in run_agent_with_streaming(user_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)
        
        st.markdown('</div>', unsafe_allow_html=True)


async def render_chat_history():
    """Rendert das Chat-Historie Interface"""
    st.markdown("<h4>📜 Chat Historie</h4>", unsafe_allow_html=True)
    
    # Suchfeld für Wildcard-Suche
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "🔍 Suche in Chat-Historie (Wildcard-Suche)", 
            placeholder="z.B. 'Wunsch-Öle' oder 'Beratung'",
            key="chat_history_search"
        )
    with col2:
        # Leerer Platz für Label-Alignment
        st.markdown("<br>", unsafe_allow_html=True)  
        search_button = st.button("Suchen", key="search_chat_history")
    
    # Chat-Historie laden (Platzhalter)
    chat_history = []  # In der echten App würde hier die DB abgefragt
    
    if not chat_history:
        if search_query:
            st.info(f"🔍 Keine Chat-Einträge für '{search_query}' gefunden.")
        else:
            st.info("📭 Noch keine Chat-Historie vorhanden.")
        return
    
    # Layout: Links die Tabelle, rechts die Details
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Chat-Einträge:**")
        st.caption("💡 Klicken Sie auf eine Zeile, um die Chat-Details anzuzeigen")
        
        # Platzhalter für Tabelle
        st.info("Chat-Historie Tabelle wird hier angezeigt")
    
    with col2:
        st.info("👈 Wählen Sie einen Chat aus der Tabelle aus, um die Details anzuzeigen.")


async def render_add_note_interface():
    """Rendert das Interface zum Hinzufügen von Notizen"""
    st.markdown("<h4>📝 Notiz hinzufügen</h4>", unsafe_allow_html=True)
    st.markdown(
        "Du kannst hier eigene Notizen, Feedback oder Empfehlungen eintragen, die sofort durchsuchbar sind."
    )

    # Platzhalter für Notiz-Interface
    st.text_input("🏷️ Überschrift")
    st.text_area("✍️ Dein Wissen")
    st.selectbox("Kategorie", ["Wissen", "Beratung", "Meeting", "Feedback", "Sonstiges"])
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.button("✅ Wissen / Notiz speichern")
    with col2:
        st.button("🧹 Eingaben leeren")


async def render_upload_interface():
    """Rendert das Interface zum Hochladen von Dokumenten"""
    st.markdown(
        """
    <h4>📎 Dateien für Wissensdatenbank hochladen</h4>
        <small>(max. 200 MB pro Datei • PDF oder TXT)</small>
    """,
        unsafe_allow_html=True,
    )

    # Platzhalter für Upload-Interface
    st.file_uploader(
        label="Dateien hochladen",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        key="uploader_placeholder"
    )


async def render_manage_docs_interface():
    """Rendert das Interface zum Anzeigen und Löschen von Dokumenten"""
    st.markdown("<h4>🗑️ Dokument / Notiz anzeigen, löschen</h4>", unsafe_allow_html=True)

    # Platzhalter für Dokumentenverwaltung
    st.info("Keine Dokumente/Notizen zur Löschung verfügbar.")


async def main():
    # Erst Daten laden, dann Header rendern
    await update_available_sources()
    render_header()

    # Robuste Initialisierung aller benötigten session_state Variablen
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sources" not in st.session_state:
        st.session_state.sources = []

    if "document_count" not in st.session_state:
        st.session_state.document_count = 0

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Schönes Tab-System wie im Original - nur mit neuer Menüstruktur
    
    # Level 1: Hauptmenüs
    tab1, tab2, tab3 = st.tabs([
        "💬 Wunsch-Öle KI Assistent",
        "➕ Wissen hinzufügen", 
        "🗑️ Dokument anzeigen / löschen"
    ])
    
    with tab1:
        # Level 2: Chat Submenüs
        chat_tab1, chat_tab2 = st.tabs(["Chat", "Chat Historie"])
        
        with chat_tab1:
            # Chat-Interface
            await render_chat_interface()
        
        with chat_tab2:
            # Chat-Historie Interface
            await render_chat_history()
    
    with tab2:
        # Level 2: Wissen hinzufügen Submenüs
        knowledge_tab1, knowledge_tab2 = st.tabs(["Notiz hinzufügen", "Dokumente hochladen"])
        
        with knowledge_tab1:
            # Notiz hinzufügen Interface
            await render_add_note_interface()
        
        with knowledge_tab2:
            # Dokumente hochladen Interface
            await render_upload_interface()
    
    with tab3:
        # Dokument anzeigen/löschen Interface
        await render_manage_docs_interface()


if __name__ == "__main__":
    asyncio.run(main())

