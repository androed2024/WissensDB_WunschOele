"""
Streamlit application for the RAG AI agent.
Virt Umgebung aktivieren: source .venv/bin/activate
Aufruf App: streamlit run app.py
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


def display_message_part(part):
    if part.part_kind == "user-prompt" and part.content:
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == "text" and part.content:
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def process_document(
    file_path: str, original_filename: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    pipeline = DocumentIngestionPipeline()
    loop = asyncio.get_event_loop()

    try:
        
        chunks = await loop.run_in_executor(
            None,
            lambda: pipeline.process_file(file_path, metadata),
        )
        if not chunks:
            return {
                "success": False,
                "file_path": file_path,
                "error": "Keine gültigen Textabschnitte gefunden",
            }

        print("\n📦 Embedding-Check")
        for i, c in enumerate(chunks):
            emb = c.get("embedding")
            text = c.get("content", "")
            print(
                f"Chunk {i+1}: Embedding: {len(emb) if emb else 0} Werte | Text: {text[:100].replace(chr(10), ' ')}..."
            )

        return {"success": True, "file_path": file_path, "chunk_count": len(chunks)}
    except Exception as e:
        import traceback

        print(f"Fehler bei der Bearbeitung des Dokuments: {str(e)}")
        print(traceback.format_exc())
        return {"success": False, "file_path": file_path, "error": str(e)}


async def run_agent_with_streaming(user_input: str):
    # Log der Original-Frage vom User
    print(f"\n🔵 [USER INPUT] Original-Frage: {user_input}")
    
    # Reset akkumulierte Treffer für neue Frage (aber nicht None setzen!)
    if hasattr(rag_agent, 'last_match'):
        rag_agent.last_match = []
        print(f"🔄 Reset: Treffer-Akkumulator für neue Frage geleert")
    
    async with rag_agent.agent.iter(
        user_input,
        deps={"kb_search": rag_agent.kb_search},
        message_history=st.session_state.messages,
    ) as run:
        async for node in run:
            if hasattr(node, "request") and isinstance(node.request, ModelRequest):
                async with node.stream(run.ctx) as request_stream:
                    async for event in request_stream:
                        if (
                            isinstance(event, PartStartEvent)
                            and event.part.part_kind == "text"
                        ):
                            yield event.part.content
                        elif isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, TextPartDelta
                        ):
                            yield event.delta.content_delta

    st.session_state.messages.extend(run.result.new_messages())


async def update_available_sources():
    try:
        response = (
            supabase_client.client.table("rag_pages").select("url, metadata").execute()
        )

        file_set = set()
        knowledge_set = set()

        for row in response.data:
            metadata = row.get("metadata", {})
            url = row.get("url", "")
            if not url:
                continue

            if metadata.get("source") == "ui_upload":
                file_set.add(url)
            elif metadata.get("source") == "manuell":
                knowledge_set.add(url)

        # 👇 Kombinieren und sortieren
        all_sources = sorted(file_set.union(knowledge_set))

        st.session_state.sources = all_sources
        st.session_state.document_count = len(file_set)
        st.session_state.knowledge_count = len(knowledge_set)

    except Exception as e:
        print(f"Fehler beim Aktualisieren der Dokumentenliste: {e}")
        for key in [
            "sources",
            "document_count",
            "knowledge_count",
            "processed_files",
            "messages",
        ]:
            if key in st.session_state:
                del st.session_state[key]


async def main():
    # Erst Daten laden, dann Header rendern
    await update_available_sources()
    render_header()

    doc_count = st.session_state.get("document_count", 0)
    note_count = st.session_state.get("knowledge_count", 0)

    # Initialisierung des Flags
    if "just_uploaded" not in st.session_state:
        st.session_state.just_uploaded = False

    # Robuste Initialisierung aller benötigten session_state Variablen
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "sources" not in st.session_state:
        st.session_state.sources = []

    if "document_count" not in st.session_state:
        try:
            st.session_state.document_count = supabase_client.count_documents()
        except Exception as e:
            print(f"Fehler beim Abrufen der Dokumentenzahl: {e}")
            st.session_state.document_count = 0

    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "💬 Wunsch-Öle KI Assistent",
            "➕ Wissen hinzufügen",
            "📄 Dokumente hochladen",
            "🗑️ Dokument anzeigen / löschen",
        ]
    )

    with tab1:
        # Vereinfachtes Layout ohne feste Header
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

        # Chat-Input Verarbeitung ZUERST
        if user_input:
            # LEERE den Chat-Bereich komplett und zeige nur die neue Unterhaltung
            st.markdown('<div class="chat-section">', unsafe_allow_html=True)
            
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                async for chunk in run_agent_with_streaming(user_input):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                # Chatbot Interface - Quellenangaben verarbeiten
                pdf_sources = defaultdict(set)
                txt_sources = defaultdict(set)
                note_sources = defaultdict(set)
                
                if hasattr(rag_agent, "last_match") and rag_agent.last_match:
                    print("--- Treffer im Retrieval ---")
                    # Score-basierte Filterung für Anzeige - identisch mit Retrieval-Schwelle
                    DISPLAY_MIN_SIM = float(os.getenv("RAG_MIN_SIM", "0.55"))
                    print(f"🎯 Anzeige-Schwellenwert: {DISPLAY_MIN_SIM} (identisch mit Retrieval)")
                    
                    # Sammle alle Matches und finde den besten Score
                    all_matches = []
                    best_score = 0
                    
                    for match in rag_agent.last_match:
                        sim = match.get("similarity", 0)
                        if sim >= DISPLAY_MIN_SIM:
                            all_matches.append((match, sim))
                            best_score = max(best_score, sim)
                    
                    # Intelligente Filterung: Wenn der beste Score deutlich höher ist,
                    # zeige nur Quellen, die nah am besten Score sind
                    RELEVANCE_GAP_THRESHOLD = 0.13  # Wenn Unterschied > 13%, filtere schwächere Quellen
                    
                    if best_score > 0:
                        # Universelle Gap-basierte Filterung (generisch für alle Anwendungsfälle)
                        gap_threshold = best_score - RELEVANCE_GAP_THRESHOLD
                        effective_threshold = max(DISPLAY_MIN_SIM, gap_threshold)
                        print(f"🎯 Intelligente Filterung: Bester Score {best_score:.3f}, Effektiver Threshold: {effective_threshold:.3f}")
                    else:
                        effective_threshold = DISPLAY_MIN_SIM
                    
                    for match, sim in all_matches:
                        # Verwende intelligenten Threshold
                        if sim < effective_threshold:
                            print(f"⚠️ Treffer gefiltert (Score {sim:.3f} < {effective_threshold:.3f} - zu weit vom besten Score entfernt)")
                            continue
                            
                        meta = match.get("metadata", {})
                        fn = meta.get("original_filename")
                        pg = meta.get("page", 1)
                        source_type = meta.get("source", "")
                        
                        if fn:
                            if source_type == "manuell":
                                note_sources[fn].add(pg)
                                print("✅ Notiz:", fn, "| Score:", sim)
                            else:
                                # Kategorisiere basierend auf Dateierweiterung
                                file_extension = fn.lower().split('.')[-1] if '.' in fn else ''
                                if file_extension == 'txt':
                                    txt_sources[fn].add(pg)
                                    print("✅ TXT-Dokument:", fn, "| Score:", sim)
                                else:
                                    pdf_sources[fn].add(pg)
                                    print("✅ PDF-Dokument:", fn, "| Seite:", pg, "| Score:", sim)

                # Bereinige die Agent-Antwort von bestehenden Quellenangaben
                import re
                
                # ULTRA-aggressive Bereinigung: Entferne alles ab dem ersten "Quelle"
                lines = full_response.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    line_lower = line.lower().strip()
                    # Stoppe bei jeder Zeile, die "quelle" enthält
                    if 'quelle' in line_lower or 'pdf öffnen' in line_lower or '.pdf' in line_lower:
                        break
                    cleaned_lines.append(line)
                
                full_response = '\n'.join(cleaned_lines).strip()
                
                # Entferne leere Zeilen am Ende
                while full_response.endswith('\n\n\n'):
                    full_response = full_response[:-1]
                
                # Prüfe ob Agent "keine Informationen" geantwortet hat (nur bei komplett leeren Antworten)
                no_info_complete_phrases = [
                    "es liegen keine informationen zu dieser frage in der wissensdatenbank vor",
                    "ich habe keine daten zu diesem thema gefunden",
                    "diese information ist nicht in der wissensdatenbank verfügbar"
                ]
                
                response_lower = full_response.lower().strip()
                # Nur unterdrücken wenn die Antwort HAUPTSÄCHLICH "keine Informationen" ist
                has_no_info_response = any(phrase in response_lower for phrase in no_info_complete_phrases)
                
                # Zusätzlich: Wenn Antwort sehr kurz ist UND "keine Informationen" enthält
                if len(response_lower) < 200:  # Kurze Antworten
                    short_no_info_phrases = [
                        "keine informationen zu dieser frage",
                        "liegen keine informationen vor"
                    ]
                    has_no_info_response = has_no_info_response or any(phrase in response_lower for phrase in short_no_info_phrases)
                
                if has_no_info_response:
                    print("🚫 Agent hat 'keine Informationen' geantwortet - unterdrücke Quellenanzeige")
                
                # Füge saubere Quellenangaben hinzu
                print(f"🔍 Debug: PDF-Quellen gefunden: {dict(pdf_sources)}")
                print(f"🔍 Debug: TXT-Quellen gefunden: {dict(txt_sources)}")
                print(f"🔍 Debug: Notiz-Quellen gefunden: {dict(note_sources)}")
                print(f"🔍 Debug: Hat 'keine Informationen' Antwort: {has_no_info_response}")
                
                # Nur Quellen anzeigen wenn Agent nicht "keine Informationen" geantwortet hat
                if (pdf_sources or txt_sources or note_sources) and not has_no_info_response:
                    full_response += "\n\n---\n"
                    
                    # Notizen zuerst anzeigen
                    for fn, pages in sorted(note_sources.items()):
                        if not pages:
                            continue
                        
                        # Hole Titel aus den Metadaten der ersten Notiz
                        try:
                            # Suche nach der Notiz in den last_match Ergebnissen
                            note_title = fn  # Fallback
                            note_filename = fn  # Fallback für Storage-Link
                            bucket = "privatedocs"  # Default
                            
                            if hasattr(rag_agent, "last_match") and rag_agent.last_match:
                                for match in rag_agent.last_match:
                                    meta = match.get("metadata", {})
                                    if (meta.get("source") == "manuell" and 
                                        (fn in meta.get("original_filename", "") or fn == meta.get("title", ""))):
                                        note_title = meta.get("title", fn)
                                        
                                        # Bestimme den korrekten Dateinamen für Storage-Link
                                        original_filename = meta.get("original_filename", fn)
                                        storage_filename = meta.get("storage_filename", original_filename)
                                        
                                        # Prüfe, ob Storage-Datei verfügbar ist
                                        has_storage = meta.get("has_storage_file", True)  # Default True für Rückwärtskompatibilität
                                        source_filter = meta.get("source_filter", "privatedocs")
                                        
                                        if has_storage and source_filter == "privatedocs":
                                            # Notiz mit Storage-Datei
                                            note_filename = storage_filename or original_filename
                                            bucket = "privatedocs"
                                        else:
                                            # Notiz ohne Storage-Datei (alte Notizen oder Upload-Fehler)
                                            note_filename = None
                                            bucket = "notes"
                                        break
                            
                            # Erstelle signed URL für die Notiz
                            if note_filename and bucket != "notes":
                                # Neue Notiz mit Storage-Datei
                                try:
                                    client = get_supabase_client()
                                    print(f"🔍 Debug: Versuche signed URL für {note_filename} in bucket {bucket}")
                                    
                                    # Erstelle signed URL für Notiz
                                    res = client.storage.from_(bucket).create_signed_url(note_filename, 3600)
                                    signed_url = res.get("signedURL", "#")
                                    
                                    print(f"🔍 Debug: Signed URL Response: {res}")
                                    print(f"🔍 Debug: Signed URL: {signed_url}")
                                    
                                    # Signed URL ist bereit für die Verwendung
                                    
                                    if signed_url and signed_url != "#":
                                        print(f"✅ Signed URL für Notiz erstellt: {signed_url[:50]}...")
                                        full_response += f"\n**📝 Notiz:** {note_title}\n"
                                        full_response += f"[📄 Notiz öffnen]({signed_url})\n"
                                    else:
                                        print(f"⚠️ Keine gültige Signed URL erhalten: {res}")
                                        full_response += f"\n**📝 Notiz:** {note_title}\n"
                                        full_response += f"(Link nicht verfügbar)\n"
                                    
                                except Exception as e:
                                    print(f"⚠️ Signed URL für Notiz fehlgeschlagen: {e}")
                                    print(f"   Exception Typ: {type(e)}")
                                    import traceback
                                    print(f"   Traceback: {traceback.format_exc()}")
                                    full_response += f"\n**📝 Notiz:** {note_title}\n"
                                    full_response += f"(Link-Fehler)\n"
                            else:
                                # Alte Notiz ohne Storage-Datei
                                print(f"⚠️ Alte Notiz ohne Storage-Link: {note_title}")
                                full_response += f"\n**📝 Notiz:** {note_title}\n"
                                full_response += f"(Nur in Datenbank gespeichert)\n"
                            
                        except Exception as e:
                            print(f"⚠️ Fehler beim Verarbeiten der Notiz-Metadaten: {e}")
                            full_response += f"\n**📝 Notiz:** {fn}\n"
                    
                    # Dann TXT-Dokumente
                    for fn, pages in sorted(txt_sources.items()):
                        if not pages:
                            continue
                        
                        # Hole die signed URL für die TXT-Datei
                        client = get_supabase_client()
                        try:
                            # Versuche verschiedene Ansätze für TXT-Dateien
                            try:
                                res = client.storage.from_("privatedocs").create_signed_url(
                                    fn, 
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
                                res = client.storage.from_("privatedocs").create_signed_url(fn, 3600)
                            
                            signed_url = res.get("signedURL")
                        except Exception as e:
                            print(f"⚠️ Fehler beim Erstellen der signed URL für TXT-Datei {fn}: {e}")
                            signed_url = "#"
                        
                        full_response += f"\n**📄 Quelle:** {fn}\n"
                        if signed_url and signed_url != "#":
                            full_response += f"[Textdatei öffnen]({signed_url})\n"
                        else:
                            full_response += "(Link nicht verfügbar)\n"
                    
                    # Dann PDF-Dokumente
                    for fn, pages in sorted(pdf_sources.items()):
                        if not pages:
                            continue
                        sorted_pages = sorted(pages)
                        page_list = ", ".join(str(pg) for pg in sorted_pages)

                        meta = {
                            "original_filename": fn,
                            "page": sorted_pages[0],
                            "source_filter": "privatedocs",
                        }
                        # Hole nur die signed URL, nicht den formatierten Text
                        client = get_supabase_client()
                        try:
                            res = client.storage.from_("privatedocs").create_signed_url(fn, 3600)
                            signed_url = res.get("signedURL")
                            if sorted_pages and signed_url:
                                signed_url += f"#page={sorted_pages[0]}"
                        except Exception as e:
                            signed_url = "#"
                        
                        full_response += f"\n**📄 Quelle:** {fn}, Seiten {page_list}\n"
                        full_response += f"[PDF öffnen]({signed_url})\n"

                # Finale Bereinigung: Entferne nur unerwünschte Agent-generierte 'Quelle:' Zeilen
                lines = full_response.split('\n')
                clean_lines = []
                for line in lines:
                    line_lower = line.lower().strip()
                    # Behalte unsere formatierten Quellen (📄, 📝), entferne nur unformatierte 'Quelle:' Zeilen
                    if line_lower.startswith('quelle:') and not ('📄' in line or '📝' in line):
                        continue
                    clean_lines.append(line)
                
                final_response = '\n'.join(clean_lines).strip()
                message_placeholder.markdown(final_response)
                
                # WICHTIG: Aktualisiere die letzte Assistant-Nachricht mit Quellenangaben
                # damit sie beim nächsten Reload korrekt angezeigt wird
                if st.session_state.messages and len(st.session_state.messages) > 0:
                    last_msg = st.session_state.messages[-1]
                    if isinstance(last_msg, ModelResponse):
                        # Ersetze den Inhalt der letzten Antwort mit der finalen Version (inkl. Quellen)
                        for i, part in enumerate(last_msg.parts):
                            if part.part_kind == "text":
                                # Erstelle neuen TextPart mit finaler Antwort
                                from pydantic_ai.messages import TextPart
                                last_msg.parts[i] = TextPart(content=final_response)
                                break
                
                # 💾 Chat-Historie speichern
                try:
                    supabase_client.save_chat_history(
                        user_name="admin",
                        question=user_input,
                        answer=final_response
                    )
                except Exception as e:
                    print(f"⚠️ Fehler beim Speichern der Chat-Historie: {e}")
                    # Fehler beim Speichern der Historie soll die App nicht zum Absturz bringen
            
            st.markdown('</div>', unsafe_allow_html=True)
        # Kein else-Block nötig - ohne Input wird einfach nichts angezeigt

    with tab2:
        st.markdown("<h4>➕ Wissen hinzufügen</h4>", unsafe_allow_html=True)
        st.markdown(
            "Du kannst hier eigene Notizen, Feedback oder Empfehlungen eintragen, die sofort durchsuchbar sind."
        )

        # Initialisierung der Eingabefelder in session_state
        for key in ["manual_title", "manual_text", "manual_source"]:
            if key not in st.session_state:
                st.session_state[key] = ""

        # Eingabefelder mit session_state
        manual_title = st.text_input(
            "🏷️ Überschrift",
            key="manual_title_input",
        )
        manual_text = st.text_area("✍️ Dein Wissen", key="manual_text_input")

        # Handle manuelle Quelle sicher
        source_options = ["Wissen", "Beratung", "Meeting", "Feedback", "Sonstiges"]
        try:
            source_index = source_options.index(st.session_state.manual_source)
        except ValueError:
            source_index = 0

        source_type = st.selectbox(
            "Kategorie",
            source_options,
            index=source_index,
            key="manual_source_input",
        )

        # Button-Reihe nebeneinander mit Columns
        col1, col2 = st.columns([3, 2])
        with col1:
            if st.button("✅ Wissen / Notiz speichern", key="save_button"):
                if not manual_title.strip() or not manual_text.strip():
                    st.warning(
                        "⚠️ Bitte gib sowohl eine Überschrift als auch einen Text ein."
                    )
                else:
                    existing = (
                        supabase_client.client.table("rag_pages")
                        .select("url")
                        .ilike("url", f"{manual_title.strip()}%")
                        .execute()
                    )
                    if existing.data:
                        st.warning(
                            f"⚠️ Ein Eintrag mit der Überschrift '{manual_title.strip()}' existiert bereits."
                        )
                    else:
                        try:
                            pipeline = DocumentIngestionPipeline()
                            tz_berlin = pytz.timezone("Europe/Berlin")
                            now_berlin = datetime.now(tz_berlin)
                            timestamp = now_berlin.strftime("%Y-%m-%d %H:%M")
                            
                            # Erstelle eine Textdatei für die Notiz (bereinige Dateinamen für Supabase)
                            import re
                            import unicodedata
                            
                            # Schritt 1: Normalisiere Unicode und entferne Akzente/Umlaute
                            normalized = unicodedata.normalize('NFD', manual_title.strip())
                            ascii_title = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
                            
                            # Schritt 2: Erlaube nur Supabase-konforme Zeichen: a-zA-Z0-9_.-
                            safe_title = re.sub(r'[^a-zA-Z0-9\s\-_.]', '', ascii_title)
                            
                            # Schritt 3: Ersetze Leerzeichen durch Unterstriche
                            safe_title = re.sub(r'\s+', '_', safe_title)
                            
                            # Schritt 4: Entferne mehrfache Unterstriche
                            safe_title = re.sub(r'_+', '_', safe_title)
                            
                            # Schritt 5: Entferne führende/trailing Unterstriche
                            safe_title = safe_title.strip('_')
                            
                            note_filename = f"{safe_title}_{now_berlin.strftime('%Y%m%d_%H%M')}.txt"
                            note_content = f"Titel: {manual_title.strip()}\nQuelle: {source_type}\nErstellt: {timestamp}\n\n{manual_text}"
                            
                            # Speichere Notiz im Storage
                            storage_success = False
                            try:
                                # UTF-8 mit BOM für bessere Browser-Kompatibilität bei deutschen Umlauten
                                note_content_bytes = '\ufeff'.encode('utf-8') + note_content.encode('utf-8')
                                
                                supabase_client.client.storage.from_("privatedocs").upload(
                                    note_filename,
                                    note_content_bytes,
                                    {
                                        "cacheControl": "3600",
                                        "x-upsert": "true",
                                        "content-type": "text/plain; charset=utf-8",
                                    },
                                )
                                print(f"✅ Notiz im Storage gespeichert: {note_filename}")
                                storage_success = True
                            except Exception as storage_error:
                                print(f"⚠️ Storage-Upload fehlgeschlagen: {storage_error}")
                                print(f"   Verwende Fallback ohne Storage-Link")
                                # Fahre trotzdem fort - Notiz wird zumindest in der DB gespeichert
                            
                            # Metadaten abhängig vom Storage-Erfolg setzen
                            if storage_success:
                                metadata = {
                                    "source": "manuell",
                                    "quelle": source_type,
                                    "title": manual_title.strip(),
                                    "upload_time": now_berlin.isoformat(),
                                    "original_filename": note_filename,  # Bereinigter Storage-Dateiname
                                    "source_filter": "privatedocs",  # Storage verfügbar
                                    "storage_filename": note_filename,
                                    "has_storage_file": True,
                                }
                            else:
                                metadata = {
                                    "source": "manuell", 
                                    "quelle": source_type,
                                    "title": manual_title.strip(),
                                    "upload_time": now_berlin.isoformat(),
                                    "original_filename": manual_title.strip(),  # Fallback: Nur Titel
                                    "source_filter": "notes",  # Kein Storage verfügbar
                                    "has_storage_file": False,
                                }
                            result = pipeline.process_text(
                                content=manual_text,
                                metadata=metadata,
                                url=manual_title.strip(),
                            )
                            st.toast(
                                "🧠 Wissen/Notizen erfolgreich gespeichert", icon="✅"
                            )
                            await update_available_sources()
                            st.session_state.manual_title = ""
                            st.session_state.manual_text = ""
                            st.session_state.manual_source = "Beratung"
                            st.rerun()
                        except Exception as e:
                            st.error(
                                f"❌ Fehler beim Speichern des Wissens/der Notiz: {e}"
                            )

        with col2:
            if st.button("🧹 Eingaben leeren", key="clear_button"):
                st.session_state.manual_title = ""
                st.session_state.manual_text = ""
                st.session_state.manual_source = "Beratung"
                st.rerun()

    with tab3:
        st.markdown(
            """
        <h4>📎 Dateien für Wissensdatenbank hochladen</h4>
            <small>(max. 200 MB pro Datei • PDF oder TXT)</small>
        """,
            unsafe_allow_html=True,
        )

        # KEIN Button und KEIN upload_clicked mehr!
        uploaded_files = st.file_uploader(
            label="Dateien hochladen",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="uploader_hidden",
            label_visibility="collapsed",
        )

        st.markdown(
            "<style>section[data-testid='stFileUploader'] label {display:none;}</style>",
            unsafe_allow_html=True,
        )

        if uploaded_files:
            # Create file list with all uploaded files (new and already processed)
            all_uploaded_files = [
                (f, f"{f.name}_{hash(f.getvalue().hex())}")
                for f in uploaded_files
            ]
            
            # Always clear old upload table when new files are selected for upload
            # This ensures clean state for each new upload session
            current_file_names = sorted([f.name for f in uploaded_files])
            last_selection = st.session_state.get("last_file_selection", [])
            
            if current_file_names != last_selection:
                # New or different file selection detected - clear old upload table
                print(f"🗑️ Neue Dateiauswahl erkannt, lösche alte Upload-Tabelle")
                if "upload_status_table" in st.session_state:
                    del st.session_state.upload_status_table
                if "just_uploaded" in st.session_state:
                    st.session_state.just_uploaded = False
                # Clear all old selection keys to prevent conflicts
                keys_to_remove = [key for key in st.session_state.keys() if key.startswith("selection_")]
                for key in keys_to_remove:
                    del st.session_state[key]
                st.session_state.last_file_selection = current_file_names
            
            # Separate new files from already processed ones
            new_files = [
                (f, file_id) for f, file_id in all_uploaded_files
                if file_id not in st.session_state.processed_files
            ]
            
            already_processed_files = [
                (f, file_id) for f, file_id in all_uploaded_files
                if file_id in st.session_state.processed_files
            ]

            # Only process if we have new files to upload
            if new_files and not st.session_state.get("currently_uploading", False):
                # Set upload in progress flag
                st.session_state.currently_uploading = True
                
                st.subheader("⏳ Upload-Status")
                
                # Clear old upload status table for new upload
                st.session_state.upload_status_table = []
                
                # Create initial table data for ALL files (new and already processed)
                table_data = []
                
                # Add new files to table
                for uploaded_file, file_id in new_files:
                    # Check file type first
                    file_ext = uploaded_file.name.lower().split('.')[-1] if '.' in uploaded_file.name else ''
                    if file_ext not in ['pdf', 'txt']:
                        table_data.append({
                            'Dateiname': uploaded_file.name,
                            'Fortschritt': 'Ungültiger Dateityp',
                            'Status': '❌ Error'
                        })
                    else:
                        table_data.append({
                            'Dateiname': uploaded_file.name,
                            'Fortschritt': '0%',
                            'Status': '⏳ Wartend'
                        })
                
                # Add already processed files to table
                for uploaded_file, file_id in already_processed_files:
                    table_data.append({
                        'Dateiname': uploaded_file.name,
                        'Fortschritt': 'Bereits in dieser Session verarbeitet',
                        'Status': '✅ Bereits hochgeladen'
                    })
                
                # Create table placeholder
                table_placeholder = st.empty()
                
                # Display initial table
                table_placeholder.table(table_data)
                
                # Process each NEW file (skip already processed ones)
                for i, (uploaded_file, file_id) in enumerate(new_files):
                    # Helper function to update table
                    def update_table_row(filename, progress, status):
                        for row in table_data:
                            if row['Dateiname'] == filename:
                                row['Fortschritt'] = progress
                                row['Status'] = status
                                break
                        table_placeholder.table(table_data)
                    
                    # Check file type first
                    file_ext = uploaded_file.name.lower().split('.')[-1] if '.' in uploaded_file.name else ''
                    if file_ext not in ['pdf', 'txt']:
                        update_table_row(uploaded_file.name, f'Nur PDF und TXT erlaubt', '❌ Error')
                        continue
                    
                    safe_filename = sanitize_filename(uploaded_file.name)
                    update_table_row(uploaded_file.name, '5%', '🔄 Prüfung...')

                    try:
                        # Check file size (200MB limit as per UI)
                        if uploaded_file.size > 200 * 1024 * 1024:  # 200MB
                            update_table_row(uploaded_file.name, 'Datei zu groß (>200MB)', '❌ Error')
                            continue
                        
                        file_bytes = uploaded_file.getvalue()
                        
                        # Check if file is empty
                        if len(file_bytes) == 0:
                            update_table_row(uploaded_file.name, 'Datei ist leer', '❌ Error')
                            continue
                            
                        file_hash = compute_file_hash(file_bytes)

                        # 🔍 Duplikatprüfung anhand Hash
                        existing_hash = (
                            supabase_client.client.table("rag_pages")
                            .select("id")
                            .eq("metadata->>file_hash", file_hash)
                            .execute()
                        )

                        if existing_hash.data:
                            update_table_row(uploaded_file.name, 'Bereits vorhanden (Hash-Duplikat)', '⚠️ Übersprungen')
                            continue

                        # ✅ Duplikatprüfung vor Upload
                        existing = (
                            supabase_client.client.table("rag_pages")
                            .select("id")
                            .eq("url", safe_filename)
                            .execute()
                        )

                        if existing.data:
                            update_table_row(uploaded_file.name, 'Bereits in Datenbank vorhanden', '⚠️ Übersprungen')
                            continue
                    
                    except Exception as e:
                        update_table_row(uploaded_file.name, f'Fehler bei Prüfung: {str(e)}', '❌ Error')
                        continue

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                    try:
                        update_table_row(uploaded_file.name, '10%', '📥 Upload startet...')

                        # Content-Type dynamisch bestimmen
                        mime_type, _ = mimetypes.guess_type(safe_filename)
                        if not mime_type:
                            mime_type = "application/octet-stream"
                        
                        # Für TXT-Dateien explizit UTF-8 Encoding setzen
                        if safe_filename.lower().endswith('.txt'):
                            mime_type = "text/plain; charset=utf-8"
                        
                        # Storage upload with error handling
                        storage_success = False
                        storage_error_msg = ""
                        
                        # Für TXT-Dateien spezielles UTF-8 Handling
                        if safe_filename.lower().endswith('.txt'):
                            # TXT-Dateien als UTF-8 Text lesen und als UTF-8 Bytes mit BOM hochladen
                            try:
                                with open(temp_file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                
                                # UTF-8 mit BOM für bessere Browser-Kompatibilität bei deutschen Umlauten (wie bei Notizen)
                                content_bytes = '\ufeff'.encode('utf-8') + content.encode('utf-8')
                                
                                print(f"🔍 Debug: TXT-Datei Upload - Content length: {len(content_bytes)} bytes (with BOM)")
                                print(f"🔍 Debug: Content-Type: {mime_type}")
                                print(f"🔍 Debug: First 50 chars: {content[:50]}...")
                                
                                supabase_client.client.storage.from_("privatedocs").upload(
                                    safe_filename,
                                    content_bytes,
                                    {
                                        "cacheControl": "3600",
                                        "x-upsert": "true",
                                        "content-type": mime_type,
                                    },
                                )
                                storage_success = True
                            except UnicodeDecodeError:
                                # Fallback: Versuche andere Encodings
                                encodings = ["latin-1", "cp1252", "ascii"]
                                content = None
                                for encoding in encodings:
                                    try:
                                        with open(temp_file_path, "r", encoding=encoding) as f:
                                            content = f.read()
                                        break
                                    except UnicodeDecodeError:
                                        continue
                                
                                try:
                                    if content:
                                        # Als UTF-8 Bytes mit BOM hochladen (konvertiert von anderem Encoding, wie bei Notizen)
                                        content_bytes = '\ufeff'.encode('utf-8') + content.encode('utf-8')
                                        print(f"🔍 Debug: TXT-Datei Fallback Upload - Content length: {len(content_bytes)} bytes (with BOM)")
                                        supabase_client.client.storage.from_("privatedocs").upload(
                                            safe_filename,
                                            content_bytes,
                                            {
                                                "cacheControl": "3600",
                                                "x-upsert": "true",
                                                "content-type": mime_type,
                                            },
                                        )
                                        storage_success = True
                                except Exception as encoding_error:
                                    storage_error_msg = f"Encoding-Upload Fehler: {str(encoding_error)}"
                                    print(f"❌ TXT Encoding-Upload Fehler für {safe_filename}: {encoding_error}")
                                    
                                if not storage_success:
                                    # Letzter Fallback: Als Bytes hochladen
                                    try:
                                        with open(temp_file_path, "rb") as f:
                                            supabase_client.client.storage.from_("privatedocs").upload(
                                                safe_filename,
                                                f.read(),
                                                {
                                                    "cacheControl": "3600",
                                                    "x-upsert": "true",
                                                    "content-type": mime_type,
                                                },
                                            )
                                            storage_success = True
                                    except Exception as final_error:
                                        storage_error_msg = f"Finaler Fallback-Fehler: {str(final_error)}"
                                        print(f"❌ Finaler Fallback-Fehler für {safe_filename}: {final_error}")
                        else:
                            # Andere Dateitypen normal als Bytes hochladen
                            try:
                                with open(temp_file_path, "rb") as f:
                                    supabase_client.client.storage.from_("privatedocs").upload(
                                        safe_filename,
                                        f.read(),  # Bytes, nicht Handle
                                        {
                                            "cacheControl": "3600",
                                            "x-upsert": "true",
                                            "content-type": mime_type,
                                        },
                                    )
                                    storage_success = True
                            except Exception as storage_error:
                                storage_error_msg = str(storage_error)
                                print(f"❌ Storage-Upload Fehler für {safe_filename}: {storage_error}")

                        if not storage_success:
                            update_table_row(uploaded_file.name, f'Storage-Fehler: {storage_error_msg}', '❌ Error')
                            continue
                            
                        update_table_row(uploaded_file.name, '30%', '📤 Dateiübertragung abgeschlossen')

                        metadata = {
                            "source": "ui_upload",
                            "upload_time": str(datetime.now()),
                            "original_filename": safe_filename,
                            "file_hash": file_hash,
                            "source_filter": "privatedocs",  # This might be missing
                        }

                        update_table_row(uploaded_file.name, '50%', '🧠 Verarbeitung läuft...')

                        result = await process_document(
                            temp_file_path, safe_filename, metadata
                        )

                        update_table_row(uploaded_file.name, '80%', '🔄 Finalisierung...')

                        if result["success"]:
                            update_table_row(uploaded_file.name, f'✅ {result["chunk_count"]} Textabschnitte', '✅ Hochgeladen')
                            st.session_state.processed_files.add(file_id)
                        else:
                            update_table_row(uploaded_file.name, f'Verarbeitungsfehler: {result["error"]}', '❌ Error')

                    except Exception as e:
                        update_table_row(uploaded_file.name, f'Unerwarteter Fehler: {str(e)}', '❌ Error')
                        print(f"❌ Unerwarteter Fehler beim Verarbeiten von {uploaded_file.name}: {e}")
                    finally:
                        if 'temp_file_path' in locals():
                            try:
                                os.unlink(temp_file_path)
                            except:
                                pass  # Ignore cleanup errors
                
                # Store table in session state for persistence
                st.session_state.upload_status_table = table_data
                
                # Final message
                successful_uploads = sum(1 for row in table_data if row['Status'] == '✅ Hochgeladen')
                already_uploaded = sum(1 for row in table_data if row['Status'] == '✅ Bereits hochgeladen')
                total_new_files = len(new_files)
                total_files = len(table_data)
                
                if total_new_files == 0 and already_uploaded > 0:
                    st.info(f"ℹ️ Alle {already_uploaded} ausgewählte(n) Datei(en) wurden bereits in dieser Session hochgeladen.")
                elif successful_uploads == total_new_files and total_new_files > 0:
                    st.success(f"🎉 Alle {total_new_files} neue(n) Datei(en) erfolgreich hochgeladen!")
                elif successful_uploads > 0:
                    st.warning(f"⚠️ {successful_uploads} von {total_new_files} neue(n) Datei(en) erfolgreich hochgeladen.")
                elif total_new_files > 0:
                    st.error(f"❌ Keine der {total_new_files} neue(n) Datei(en) konnten hochgeladen werden.")

                # Reset upload flags
                st.session_state.just_uploaded = True
                st.session_state.currently_uploading = False
                
                # Aktualisiere Quellen explizit nach Upload
                await update_available_sources()
                print(f"🔄 Nach Upload: {st.session_state.get('document_count', 0)} Dokumente, {st.session_state.get('knowledge_count', 0)} Notizen")
                
                # Seite neu laden damit Header mit aktualisierten Zählern angezeigt wird
                st.rerun()

            elif already_processed_files and not new_files:
                st.info("Alle Dateien wurden bereits verarbeitet")
        
        # Display persistent upload status table if it exists (but not during active upload)
        elif ("upload_status_table" in st.session_state and 
              st.session_state.upload_status_table and 
              not st.session_state.get("currently_uploading", False)):
            st.subheader("📊 Letzter Upload-Status")
            st.table(st.session_state.upload_status_table)
            
            if st.button("🧹 Upload-Historie löschen", key="clear_upload_history"):
                del st.session_state.upload_status_table
                # Reset flags when clearing history
                if "just_uploaded" in st.session_state:
                    st.session_state.just_uploaded = False
                st.rerun()

        st.markdown(
            "<hr style='margin-top: 6px; margin-bottom: 6px;'>", unsafe_allow_html=True
        )

    with tab4:
        st.markdown("<h4>🗑️ Dokument / Notiz löschen</h4>", unsafe_allow_html=True)
        
        # Custom CSS for better text readability in preview areas
        st.markdown("""
        <style>
        /* Improve readability of disabled text areas - use very light background and black text */
        .stTextArea textarea[disabled] {
            color: #000000 !important;
            background-color: #f8f9fa !important;
            opacity: 1 !important;
            font-family: 'Source Code Pro', monospace !important;
            line-height: 1.5 !important;
            border: 1px solid #dee2e6 !important;
            -webkit-text-fill-color: #000000 !important;
        }
        
        /* Force black text color with higher specificity */
        div[data-testid="stTextArea"] textarea[disabled] {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
        }
        
        /* Also apply to regular text areas for consistency */
        .stTextArea textarea {
            line-height: 1.5 !important;
        }
        </style>
        """, unsafe_allow_html=True)

        if st.session_state.sources:
            delete_filename = st.selectbox(
                "Dokument/Notiz selektieren", st.session_state.sources
            )

            # Vorschau anzeigen
            st.markdown("### 📄 Vorschau")

            # Metadaten aus Supabase holen
            try:
                res = (
                    supabase_client.client.table("rag_pages")
                    .select("content", "metadata")
                    .eq("url", delete_filename)
                    .limit(1)
                    .execute()
                )

                if res.data:
                    entry = res.data[0]
                    content = entry.get("content", "")
                    metadata = entry.get("metadata", {})
                    source = metadata.get("source", "")

                    if source == "manuell":
                        # Put title and source side by side
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**Titel:** {metadata.get('title', 'Unbekannt')}")
                        with col2:
                            st.markdown(f"**Quelle:** {metadata.get('quelle', '–')}")
                        
                        # Use text_area with proper line breaks and scrolling like text files
                        st.text_area(
                            "Notizinhalt", 
                            content, 
                            height=400, 
                            disabled=True,
                            key=f"note_preview_{hash(delete_filename)}"
                        )
                    else:
                        # Unterscheidung zwischen PDF und TXT Dateien
                        original_filename = metadata.get("original_filename", "")
                        file_extension = metadata.get("file_extension", "").lower()
                        
                        # Fallback: Determine file extension from filename if not in metadata
                        if not file_extension and original_filename:
                            if '.' in original_filename:
                                file_extension = '.' + original_filename.lower().split('.')[-1]
                            else:
                                file_extension = ""
                        
                        if file_extension == ".pdf":
                            # Original PDF anzeigen
                            try:
                                st.markdown("**📄 Original-PDF Vorschau:**")
                                
                                # Create signed URL dynamically for PDF preview
                                client = get_supabase_client()
                                try:
                                    res = client.storage.from_("privatedocs").create_signed_url(original_filename, 3600)
                                    signed_url = res.get("signedURL")
                                    
                                    if signed_url and signed_url != "#":
                                        st.markdown("**PDF-Inhalt:**")
                                        
                                        # Try multiple PDF viewer approaches for maximum compatibility
                                        
                                        # Approach 1: Try Mozilla PDF.js viewer first (most reliable)
                                        pdfjs_url = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={signed_url}"
                                        
                                        st.components.v1.html(
                                            f"""
                                            <div style="width: 100%; height: 800px; border: 1px solid #ccc; border-radius: 6px; margin-top: 10px; position: relative;">
                                                <iframe 
                                                    src="{pdfjs_url}" 
                                                    width="100%" 
                                                    height="100%" 
                                                    style="border: none; border-radius: 6px;"
                                                    frameborder="0"
                                                    onload="console.log('PDF.js loaded successfully')"
                                                    onerror="console.error('PDF.js failed to load'); this.style.display='none'; document.getElementById('fallback-{hash(signed_url)}').style.display='block';">
                                                </iframe>
                                                
                                                <!-- Fallback for when PDF.js doesn't work -->
                                                <div id="fallback-{hash(signed_url)}" style="display: none; padding: 20px; text-align: center; height: 100%;">
                                                    <div style="margin-top: 200px;">
                                                        <h3>PDF-Vorschau nicht verfügbar</h3>
                                                        <p>Der PDF-Viewer konnte nicht geladen werden.</p>
                                                        <a href="{signed_url}" target="_blank" 
                                                           style="display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; font-size: 16px;">
                                                            📄 PDF in neuem Tab öffnen
                                                        </a>
                                                    </div>
                                                </div>
                                                
                                                <div style="position: absolute; bottom: 0; left: 0; right: 0; text-align: center; padding: 5px; background: rgba(248,249,250,0.9); border-top: 1px solid #dee2e6;">
                                                    <small>
                                                        <a href="{signed_url}" target="_blank" style="color: #007bff; text-decoration: none;">
                                                            📄 PDF in neuem Tab öffnen
                                                        </a>
                                                    </small>
                                                </div>
                                            </div>
                                            """,
                                            height=820,
                                        )
                                    else:
                                        st.warning("⚠️ PDF-Vorschau konnte nicht geladen werden. Signed URL nicht verfügbar.")
                                        st.info(f"Debug: Response: {res}")
                                        
                                except Exception as url_error:
                                    st.error(f"❌ Fehler beim Erstellen der PDF-Vorschau URL: {url_error}")
                                    st.info("Versuche alternative Anzeige...")
                                    
                                    # Fallback: Show text content if available
                                    if content and content.strip():
                                        st.markdown("**Extrahierter Text-Inhalt:**")
                                        st.text_area(
                                            "PDF Textinhalt", 
                                            content, 
                                            height=400, 
                                            disabled=True,
                                            key=f"pdf_text_preview_{hash(delete_filename)}"
                                        )
                                    else:
                                        st.warning("Keine PDF-Vorschau oder Textinhalt verfügbar.")
                                        
                            except Exception as e:
                                st.error(f"❌ Fehler beim Laden der PDF-Vorschau: {e}")
                                # Show text content as fallback
                                if content and content.strip():
                                    st.markdown("**Extrahierter Text-Inhalt (Fallback):**")
                                    st.text_area(
                                        "PDF Textinhalt", 
                                        content, 
                                        height=400, 
                                        disabled=True,
                                        key=f"pdf_fallback_preview_{hash(delete_filename)}"
                                    )
                        elif file_extension == ".txt":
                            # TXT-Datei Vorschau
                            try:
                                st.markdown("**📄 Text-Datei Vorschau:**")
                                # Put filename, file size and chunks on same line
                                col1, col2, col3 = st.columns([2, 1, 1])
                                with col1:
                                    st.markdown(f"**Dateiname:** {original_filename}")
                                with col2:
                                    st.markdown(f"**Dateigröße:** {metadata.get('file_size_bytes', 0)} Bytes")
                                with col3:
                                    st.markdown(f"**Chunks:** {metadata.get('chunk_count', 1)}")
                                
                                st.text_area(
                                    "Dateiinhalt", 
                                    content, 
                                    height=400, 
                                    disabled=True,
                                    key=f"txt_preview_{hash(delete_filename)}"
                                )
                            except Exception as e:
                                st.error(f"Fehler beim Laden der TXT-Vorschau: {e}")
                        else:
                            # Fallback für andere Dateitypen
                            st.markdown(f"**📄 Dokument Vorschau ({original_filename}):**")
                            st.markdown(f"**Dateityp:** {file_extension if file_extension else 'Unbekannt'}")
                            st.markdown("**Inhalt:**")
                            st.text_area(
                                "Dokumentinhalt", 
                                content, 
                                height=400, 
                                disabled=True,
                                key=f"doc_preview_{hash(delete_filename)}"
                            )
                else:
                    st.info("Keine Vorschau verfügbar.")

            except Exception as e:
                st.error(f"Fehler beim Laden der Vorschau: {e}")

            if st.button("Ausgewähltes Dokument/Notiz löschen"):
                st.write("Dateiname zur Löschung:", delete_filename)

                storage_deleted = db_deleted = False

                # Erst Datenbank, dann Storage löschen (umgekehrte Reihenfolge)
                try:
                    print(f"🗑️ Lösche Datenbankeinträge für: {delete_filename}")
                    deleted_count = supabase_client.delete_documents_by_filename(
                        delete_filename
                    )
                    st.code(
                        f"🩨 SQL-Delete für '{delete_filename}' – {deleted_count} Einträge entfernt."
                    )
                    if deleted_count > 0:
                        db_deleted = True
                        print(f"✅ {deleted_count} Datenbankeinträge erfolgreich gelöscht")
                    else:
                        print(f"⚠️ Keine Datenbankeinträge für {delete_filename} gefunden")
                except Exception as e:
                    st.error(f"Datenbank-Löschung fehlgeschlagen: {e}")
                    print(f"❌ Datenbank-Löschung fehlgeschlagen: {e}")
                    db_deleted = False

                try:
                    print(f"🗑️ Lösche Storage-Datei: {delete_filename}")
                    supabase_client.client.storage.from_("privatedocs").remove(
                        [delete_filename]
                    )
                    storage_deleted = True
                    print(f"✅ Storage-Datei erfolgreich gelöscht")
                except Exception as e:
                    st.error(f"Löschen aus dem Speicher fehlgeschlagen: {e}")
                    print(f"❌ Storage-Löschung fehlgeschlagen: {e}")
                    
                # Zusätzliche Verifikation: Prüfe ob wirklich gelöscht
                try:
                    print(f"🔍 Verifikation: Prüfe ob {delete_filename} wirklich gelöscht wurde...")
                    verify_result = supabase_client.client.table("rag_pages").select("id,url").eq("url", delete_filename).execute()
                    remaining_entries = len(verify_result.data or [])
                    if remaining_entries > 0:
                        print(f"⚠️ WARNUNG: {remaining_entries} Einträge für {delete_filename} sind noch in der Datenbank!")
                        st.warning(f"⚠️ {remaining_entries} Einträge sind noch in der Datenbank vorhanden!")
                        # Versuche nochmal zu löschen
                        print("🔄 Versuche erneute Löschung...")
                        retry_deleted = supabase_client.delete_documents_by_filename(delete_filename)
                        print(f"🔄 Zweiter Löschversuch: {retry_deleted} Einträge entfernt")
                    else:
                        print(f"✅ Verifikation erfolgreich: Keine Einträge für {delete_filename} gefunden")
                except Exception as e:
                    print(f"❌ Verifikation fehlgeschlagen: {e}")
                    st.error(f"Verifikation fehlgeschlagen: {e}")

                if storage_deleted and db_deleted:
                    st.success("✅ Vollständig gelöscht.")
                    # Remove from processed files list so it can be re-uploaded
                    if "processed_files" in st.session_state:
                        # Remove all entries that match this filename
                        files_to_remove = [f for f in st.session_state.processed_files if delete_filename in f]
                        for file_to_remove in files_to_remove:
                            st.session_state.processed_files.discard(file_to_remove)
                            print(f"🧹 Entfernt aus processed_files: {file_to_remove}")
                elif storage_deleted and not db_deleted:
                    st.warning(
                        "⚠️ Dokument/Notiz im Storage gelöscht, aber kein Eintrag in der Datenbank gefunden."
                    )
                elif not storage_deleted and db_deleted:
                    st.warning(
                        "⚠️ Datenbankeinträge gelöscht, aber Dokument/Notiz im Storage konnte nicht entfernt werden."
                    )
                else:
                    st.error(
                        "❌ Weder Dokument/Notiz noch Datenbankeinträge konnten gelöscht werden."
                    )

                # Cache leeren und Quellen aktualisieren
                print("🔄 Aktualisiere verfügbare Quellen nach Löschung...")
                await update_available_sources()
                
                # Zusätzlich: Session State Cache leeren
                cache_keys_to_clear = ["sources", "document_count", "knowledge_count"]
                for key in cache_keys_to_clear:
                    if key in st.session_state:
                        old_value = st.session_state[key]
                        del st.session_state[key]
                        print(f"🧹 Cache geleert: {key} (war: {old_value})")
                
                print("🔄 Seite wird neu geladen...")
                st.rerun()

        else:
            st.info("Keine Dokumente/Notizen zur Löschung verfügbar.")


if __name__ == "__main__":
    asyncio.run(main())
