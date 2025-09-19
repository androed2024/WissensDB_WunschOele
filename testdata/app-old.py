"""
Streamlit application for the RAG AI agent with restructured menu system.
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
        <div class="header-flex" style="display: flex; justify-content: space-between; align-items: center; padding-top: 0.5rem; padding-bottom: 0.5rem;">
            <div class="header-title-wrap" style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{encoded}" alt="Logo" style="height: 42px; margin-right: 14px;">
                <span style="font-size: 22px; font-weight: 600;">Wunsch-Öle Wissens Agent</span>
                <span style="color: #007BFF; font-size: 14px; margin-left: 12px;">🔧 Version: {app_version}</span>
            </div>
            <div style="font-size: 14px;">
                📄 Dokumente: {doc_count} &nbsp;&nbsp;&nbsp; 🧠 Notizen: {note_count}
            </div>
        </div>
        <hr style="margin-top: 0.4rem; margin-bottom: 0.8rem;">
        """,
        unsafe_allow_html=True,
    )

supabase_client = SupabaseClient()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Menu state initialization
if "selected_main_menu" not in st.session_state:
    st.session_state.selected_main_menu = "💬 Wunsch-Öle KI Assistent"

if "selected_sub_menu" not in st.session_state:
    st.session_state.selected_sub_menu = None

if "chat_history_search" not in st.session_state:
    st.session_state.chat_history_search = ""

if "selected_chat_id" not in st.session_state:
    st.session_state.selected_chat_id = None


def display_message_part(part):
    if part.part_kind == "user-prompt" and part.content:
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == "text" and part.content:
        with st.chat_message("assistant"):
            st.markdown(part.content)


from typing import Callable, Optional

# Type alias for phase callback
PhaseCB = Callable[[str, int, int], None]

async def process_document(
    file_path: str, 
    safe_filename: str, 
    metadata: Dict[str, Any],
    on_phase: Optional[PhaseCB] = None
) -> Dict[str, Any]:
    """
    Asynchrone Dokumentenverarbeitung mit Phasen-Callbacks.
    
    Args:
        file_path: Pfad zur zu verarbeitenden Datei
        safe_filename: Sicherer Dateiname für die Speicherung
        metadata: Metadaten für das Dokument
        on_phase: Optional callback für Phasen-Updates (phase, processed, total)
        
    Returns:
        Dict mit success, chunk_count und ggf. error
    """
    pipeline = DocumentIngestionPipeline()
    loop = asyncio.get_event_loop()

    try:
        # P4: Finalisierung wird am Ende aufgerufen
        def wrapped_on_phase(phase: str, processed: int, total: int):
            if on_phase:
                try:
                    on_phase(phase, processed, total)
                except Exception as e:
                    print(f"⚠️ Fehler bei Progress-Update ({phase}): {e}")
                    # Continue processing even if UI update fails
        
        chunks = await loop.run_in_executor(
            None,
            lambda: pipeline.process_file(file_path, metadata, on_phase=wrapped_on_phase),
        )
        
        if not chunks:
            return {
                "success": False,
                "file_path": file_path,
                "error": "Keine gültigen Textabschnitte gefunden",
            }

        # P4: Finalisierung - Sign URLs, Checks, Cache Warmup
        if on_phase:
            on_phase("finalize", 0, 1)
            
        print("\n📦 Embedding-Check")
        for i, c in enumerate(chunks):
            emb = c.get("embedding")
            text = c.get("content", "")
            print(
                f"Chunk {i+1}: Embedding: {len(emb) if emb else 0} Werte | Text: {text[:100].replace(chr(10), ' ')}..."
            )
        
        # Finalisierung abgeschlossen
        if on_phase:
            on_phase("finalize", 1, 1)

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


def get_chat_history(search_term: str = "") -> List[Dict]:
    """Holt Chat-Historie aus Supabase mit optionaler Wildcard-Suche"""
    try:
        query = supabase_client.client.table("chat_history").select("*")
        
        if search_term.strip():
            # Wildcard-Suche auf Frage-Feld
            search_pattern = f"%{search_term}%"
            query = query.ilike("question", search_pattern)
        
        response = query.order("created_at", desc=True).execute()
        return response.data or []
    except Exception as e:
        print(f"Fehler beim Abrufen der Chat-Historie: {e}")
        return []


def _get_query_params() -> dict:
    try:
        # Newer Streamlit
        return dict(st.query_params)
    except Exception:
        # Fallback for older versions
        try:
            return {k: v[0] if isinstance(v, list) else v for k, v in st.experimental_get_query_params().items()}
        except Exception:
            return {}


def _set_query_params(**params) -> None:
    try:
        # Newer Streamlit API
        st.query_params.clear()
        for k, v in params.items():
            if v is not None:
                st.query_params[k] = v
    except Exception:
        # Fallback API
        st.experimental_set_query_params(**{k: v for k, v in params.items() if v is not None})


def render_main_menu():
    """Rendert das Hauptmenü als reine Text-Navigation ohne URL-Wechsel."""
    st.markdown(
        """
        <style>
        /* Global compact layout */
        .block-container { padding-top: 0.2rem !important; padding-bottom: 0.4rem !important; }
        .header-flex { margin-bottom: 0.2rem !important; }
        hr { margin-top: 0.2rem !important; margin-bottom: 0.2rem !important; }

        .main-menu-scope div[data-baseweb="radio"] > div {
            display: flex; flex-direction: row; gap: 20px;
            border-bottom: 1px solid #e9ecef; padding: 2px 0 4px 0; margin: 0 0 4px 0;
        }
        .main-menu-scope div[data-baseweb="radio"] label {
            background: transparent; border: none; padding: 2px 4px; border-radius: 4px;
            cursor: pointer; font-size: 16px; font-weight: 500; color: #495057;
        }
        .main-menu-scope div[data-baseweb="radio"] label:hover { color: #dc3545; }
        /* Hide radio circles robustly */
        .main-menu-scope div[data-baseweb="radio"] label > div:first-child,
        .main-menu-scope div[data-baseweb="radio"] label [data-baseweb],
        .main-menu-scope div[data-baseweb="radio"] label svg { display: none !important; }
        .main-menu-scope div[data-baseweb="radio"] label[data-checked="true"] {
            color: #dc3545; border-bottom: 2px solid #dc3545; font-weight: 600;
        }
        
        .submenu-scope div[data-baseweb="radio"] > div {
            display: flex; flex-direction: row; gap: 14px; margin: 4px 0 6px 8px;
        }
        .submenu-scope div[data-baseweb="radio"] label {
            background: transparent; border: none; padding: 2px 3px; border-radius: 3px;
            cursor: pointer; font-size: 14px; color: #6c757d;
        }
        .submenu-scope div[data-baseweb="radio"] label:hover { color: #dc3545; text-decoration: underline; }
        .submenu-scope div[data-baseweb="radio"] label > div:first-child,
        .submenu-scope div[data-baseweb="radio"] label [data-baseweb],
        .submenu-scope div[data-baseweb="radio"] label svg { display: none !important; }
        .submenu-scope div[data-baseweb="radio"] label[data-checked="true"] {
            color: #dc3545; font-weight: 500; text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    options = [
        "💬 Wunsch-Öle KI Assistent",
        "➕ Wissen hinzufügen",
        "🗑️ Dokument anzeigen / löschen",
    ]

    current = st.session_state.get("selected_main_menu", options[0])
    st.markdown("<div class='main-menu-scope'>", unsafe_allow_html=True)
    selected = st.radio(
        "Hauptmenü",
        options,
        index=options.index(current) if current in options else 0,
        horizontal=True,
        label_visibility="collapsed",
        key="main_menu_text_radio",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if selected != st.session_state.get("selected_main_menu"):
        st.session_state.selected_main_menu = selected
        st.session_state.selected_sub_menu = None
        st.rerun()


async def process_chat_input(user_input: str):
    """Process chat input with streaming response"""
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Display assistant response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        async for chunk in run_agent_with_streaming(user_input):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        # Process sources
        pdf_sources = defaultdict(set)
        txt_sources = defaultdict(set)
        note_sources = defaultdict(set)
        
        if hasattr(rag_agent, "last_match") and rag_agent.last_match:
            DISPLAY_MIN_SIM = float(os.getenv("RAG_MIN_SIM", "0.55"))
            
            # Sammle alle Matches und finde den besten Score
            all_matches = []
            best_score = 0
            
            for match in rag_agent.last_match:
                sim = match.get("similarity", 0)
                if sim >= DISPLAY_MIN_SIM:
                    all_matches.append((match, sim))
                    best_score = max(best_score, sim)
            
            # Intelligente Filterung
            RELEVANCE_GAP_THRESHOLD = 0.13
            if best_score > 0:
                gap_threshold = best_score - RELEVANCE_GAP_THRESHOLD
                effective_threshold = max(DISPLAY_MIN_SIM, gap_threshold)
            else:
                effective_threshold = DISPLAY_MIN_SIM
            
            for match, sim in all_matches:
                if sim < effective_threshold:
                    continue
                    
                meta = match.get("metadata", {})
                fn = meta.get("original_filename")
                pg = meta.get("page", 1)
                source_type = meta.get("source", "")
                
                if fn:
                    if source_type == "manuell":
                        note_sources[fn].add(pg)
                    else:
                        file_extension = fn.lower().split('.')[-1] if '.' in fn else ''
                        if file_extension == 'txt':
                            txt_sources[fn].add(pg)
                        else:
                            pdf_sources[fn].add(pg)

        # Clean response from existing source references
        lines = full_response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line_lower = line.lower().strip()
            if 'quelle' in line_lower or 'pdf öffnen' in line_lower or '.pdf' in line_lower:
                break
            cleaned_lines.append(line)
        
        full_response = '\n'.join(cleaned_lines).strip()
        
        # Check if agent responded with "no information"
        no_info_phrases = [
            "es liegen keine informationen zu dieser frage in der wissensdatenbank vor",
            "ich habe keine daten zu diesem thema gefunden",
            "diese information ist nicht in der wissensdatenbank verfügbar"
        ]
        
        response_lower = full_response.lower().strip()
        has_no_info_response = any(phrase in response_lower for phrase in no_info_phrases)
        
        if len(response_lower) < 200:
            short_no_info_phrases = [
                "keine informationen zu dieser frage",
                "liegen keine informationen vor"
            ]
            has_no_info_response = has_no_info_response or any(phrase in response_lower for phrase in short_no_info_phrases)

        # Add clean source references
        if (pdf_sources or txt_sources or note_sources) and not has_no_info_response:
            full_response += "\n\n---\n"
            
            # Add note sources
            for fn, pages in sorted(note_sources.items()):
                if not pages:
                    continue
                
                # Get note title from metadata
                note_title = fn
                if hasattr(rag_agent, "last_match") and rag_agent.last_match:
                    for match in rag_agent.last_match:
                        meta = match.get("metadata", {})
                        if (meta.get("source") == "manuell" and 
                            (fn in meta.get("original_filename", "") or fn == meta.get("title", ""))):
                            note_title = meta.get("title", fn)
                            break
                
                full_response += f"\n**📝 Notiz:** {note_title}\n"
            
            # Add txt sources
            for fn, pages in sorted(txt_sources.items()):
                if not pages:
                    continue
                full_response += f"\n**📄 Quelle:** {fn}\n"
            
            # Add PDF sources
            for fn, pages in sorted(pdf_sources.items()):
                if not pages:
                    continue
                sorted_pages = sorted(pages)
                page_list = ", ".join(str(pg) for pg in sorted_pages)
                full_response += f"\n**📄 Quelle:** {fn}, Seiten {page_list}\n"

        # Final cleanup
        lines = full_response.split('\n')
        clean_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            if line_lower.startswith('quelle:') and not ('📄' in line or '📝' in line):
                continue
            clean_lines.append(line)
        
        final_response = '\n'.join(clean_lines).strip()
        message_placeholder.markdown(final_response)
        
        # Update last assistant message
        if st.session_state.messages and len(st.session_state.messages) > 0:
            last_msg = st.session_state.messages[-1]
            if isinstance(last_msg, ModelResponse):
                for i, part in enumerate(last_msg.parts):
                    if part.part_kind == "text":
                        from pydantic_ai.messages import TextPart
                        last_msg.parts[i] = TextPart(content=final_response)
                        break
        
        # Save chat history
        try:
            supabase_client.save_chat_history(
                user_name="admin",
                question=user_input,
                answer=final_response
            )
        except Exception as e:
            print(f"⚠️ Fehler beim Speichern der Chat-Historie: {e}")


def render_chat_interface():
    """Rendert die Chat-Schnittstelle (Input direkt unter dem Untermenü)."""
    # Oberer Eingabe-Bereich (nicht sticky)
    with st.container():
        col_a, col_b = st.columns([5,1])
        with col_a:
            question = st.text_input("Stelle eine Frage zu den Dokumenten...", key="top_question_input")
        with col_b:
            send = st.button("Senden", key="top_question_send")
        if send and question.strip():
            st.session_state.pending_chat_input = question.strip()
            st.rerun()

    # Optional: bisherige sticky Chat-Input ganz unten deaktiviert
    # (bewusst entfernt, um Doppelung zu vermeiden)


def render_chat_history():
    """Rendert die Chat-Historie mit Suchfunktion"""
    st.markdown("### 📜 Chat Historie")
    
    # Suchfeld
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "🔍 Suche in Fragen (Wildcard-Suche)", 
            value=st.session_state.chat_history_search,
            key="chat_history_search_input"
        )
    with col2:
        if st.button("🔍 Suchen", key="search_chat_history"):
            st.session_state.chat_history_search = search_term
            st.rerun()
    
    # Chat-Historie abrufen
    chat_history = get_chat_history(st.session_state.chat_history_search)
    
    if not chat_history:
        if st.session_state.chat_history_search:
            st.info(f"Keine Chats gefunden für Suchbegriff: '{st.session_state.chat_history_search}'")
        else:
            st.info("Keine Chat-Historie verfügbar.")
        return
    
    # Zwei-Spalten Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Chat-Liste:**")
        for chat in chat_history:
            chat_id = chat.get("id")
            question = chat.get("question", "")
            created_at = chat.get("created_at", "")
            user_name = chat.get("user", "Unbekannt")
            
            # Kürze die Frage für die Anzeige
            display_question = question[:60] + "..." if len(question) > 60 else question
            
            # Formatiere Datum
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                formatted_date = dt.strftime("%d.%m.%Y %H:%M")
            except:
                formatted_date = created_at
            
            # Button für jeden Chat
            if st.button(
                f"📅 {formatted_date}\n👤 {user_name}\n❓ {display_question}",
                key=f"chat_{chat_id}",
                use_container_width=True
            ):
                st.session_state.selected_chat_id = chat_id
                st.rerun()
    
    with col2:
        st.markdown("**Chat-Details:**")
        if st.session_state.selected_chat_id:
            # Finde den ausgewählten Chat
            selected_chat = next(
                (chat for chat in chat_history if chat.get("id") == st.session_state.selected_chat_id),
                None
            )
            
            if selected_chat:
                question = selected_chat.get("question", "")
                answer = selected_chat.get("answer", "")
                created_at = selected_chat.get("created_at", "")
                user_name = selected_chat.get("user", "Unbekannt")
                
                # Formatiere Datum
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    formatted_date = dt.strftime("%d.%m.%Y um %H:%M Uhr")
                except:
                    formatted_date = created_at
                
                # Header
                st.markdown(f"**Datum:** {formatted_date}")
                st.markdown(f"**Benutzer:** {user_name}")
                st.markdown("---")
                
                # Frage und Antwort
                with st.chat_message("user"):
                    st.markdown(question)
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
            else:
                st.info("Chat nicht gefunden.")
        else:
            st.info("Wählen Sie einen Chat aus der Liste aus, um die Details zu sehen.")


def render_add_note():
    """Rendert das Notiz-Hinzufügen Interface"""
    st.markdown("### ✍️ Notiz hinzufügen")
    st.markdown(
        "Du kannst hier eigene Notizen, Feedback oder Empfehlungen eintragen, die sofort durchsuchbar sind."
    )

    # Eingabefelder
    manual_title = st.text_input("🏷️ Überschrift", key="manual_title_input")
    manual_text = st.text_area("✍️ Dein Wissen", key="manual_text_input")

    source_options = ["Wissen", "Beratung", "Meeting", "Feedback", "Sonstiges"]
    source_type = st.selectbox("Kategorie", source_options, key="manual_source_input")

    # Buttons
    col1, col2 = st.columns([3, 2])
    with col1:
        if st.button("✅ Wissen / Notiz speichern", key="save_button"):
            if not manual_title.strip() or not manual_text.strip():
                st.warning("⚠️ Bitte gib sowohl eine Überschrift als auch einen Text ein.")
            else:
                existing = (
                    supabase_client.client.table("rag_pages")
                    .select("url")
                    .ilike("url", f"{manual_title.strip()}%")
                    .execute()
                )
                if existing.data:
                    st.warning(f"⚠️ Ein Eintrag mit der Überschrift '{manual_title.strip()}' existiert bereits.")
                else:
                    try:
                        pipeline = DocumentIngestionPipeline()
                        tz_berlin = pytz.timezone("Europe/Berlin")
                        now_berlin = datetime.now(tz_berlin)
                        timestamp = now_berlin.strftime("%Y-%m-%d %H:%M")
                        
                        # Create safe filename
                        normalized = unicodedata.normalize('NFD', manual_title.strip())
                        ascii_title = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
                        safe_title = re.sub(r'[^a-zA-Z0-9\s\-_.]', '', ascii_title)
                        safe_title = re.sub(r'\s+', '_', safe_title)
                        safe_title = re.sub(r'_+', '_', safe_title)
                        safe_title = safe_title.strip('_')
                        
                        note_filename = f"{safe_title}_{now_berlin.strftime('%Y%m%d_%H%M')}.txt"
                        note_content = f"Titel: {manual_title.strip()}\nQuelle: {source_type}\nErstellt: {timestamp}\n\n{manual_text}"
                        
                        # Save to storage
                        storage_success = False
                        try:
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
                            storage_success = True
                        except Exception as storage_error:
                            print(f"⚠️ Storage-Upload fehlgeschlagen: {storage_error}")
                        
                        # Set metadata
                        if storage_success:
                            metadata = {
                                "source": "manuell",
                                "quelle": source_type,
                                "title": manual_title.strip(),
                                "upload_time": now_berlin.isoformat(),
                                "original_filename": note_filename,
                                "source_filter": "privatedocs",
                                "storage_filename": note_filename,
                                "has_storage_file": True,
                            }
                        else:
                            metadata = {
                                "source": "manuell", 
                                "quelle": source_type,
                                "title": manual_title.strip(),
                                "upload_time": now_berlin.isoformat(),
                                "original_filename": manual_title.strip(),
                                "source_filter": "notes",
                                "has_storage_file": False,
                            }
                        
                        result = pipeline.process_text(
                            content=manual_text,
                            metadata=metadata,
                            url=manual_title.strip(),
                        )
                        
                        st.toast("🧠 Wissen/Notizen erfolgreich gespeichert", icon="✅")
                        st.session_state.need_source_update = True
                        
                        # Clear inputs
                        st.session_state.manual_title_input = ""
                        st.session_state.manual_text_input = ""
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Fehler beim Speichern des Wissens/der Notiz: {e}")

    with col2:
        if st.button("🧹 Eingaben leeren", key="clear_button"):
            st.session_state.manual_title_input = ""
            st.session_state.manual_text_input = ""
            st.rerun()


def render_document_upload():
    """Rendert das Dokument-Upload Interface"""
    st.markdown("### 📎 Dokumente hochladen")
    st.markdown(
        """
    <small>(max. 200 MB pro Datei • PDF oder TXT)</small>
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
            if "upload_just_completed" in st.session_state:
                st.session_state.upload_just_completed = False
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
                        
                    # Initialize ProgressTracker after upload completion
                    tracker = ProgressTracker(
                        filename=uploaded_file.name,
                        update_ui=lambda fn, pct, status: update_table_row(fn, pct, status),
                        file_size_bytes=uploaded_file.size
                    )
                    
                    # Mark upload phase as completed
                    tracker.set_total("upload", 1)
                    tracker.tick("upload", 1, status="📤 Dateiübertragung abgeschlossen")

                    # Pre-estimation for logging (optional)
                    try:
                        total_chars = count_text_chars(temp_file_path, mime_type)
                        est_chunks = estimate_total_chunks(total_chars, CHUNK_SIZE, CHUNK_OVERLAP)
                        print(f"📈 Geschätzte Chunks für {uploaded_file.name}: {est_chunks} (basierend auf {total_chars} Zeichen)")
                    except Exception as e:
                        print(f"⚠️ Fehler bei Chunk-Schätzung für {uploaded_file.name}: {e}")
                        est_chunks = None

                    metadata = {
                        "source": "ui_upload",
                        "upload_time": str(datetime.now()),
                        "original_filename": safe_filename,
                        "file_hash": file_hash,
                        "source_filter": "privatedocs",
                    }

                    # Store progress state (thread-safe) - UI updates happen in main thread
                    progress_state = {
                        "phase": "upload",
                        "processed": 0,
                        "total": 1,
                        "last_update": 0
                    }
                    
                    def on_phase(phase: str, processed: int, total: int):
                        try:
                            # Just store state - no UI updates from thread
                            progress_state["phase"] = phase
                            progress_state["processed"] = processed
                            progress_state["total"] = total
                            progress_state["last_update"] = time.time()
                            print(f"📈 {phase}: {processed}/{total}")
                        except Exception as e:
                            print(f"⚠️ Progress-State Fehler ({phase}): {e}")

                    # Start processing with real progress tracking
                    update_table_row(uploaded_file.name, '35%', '🧠 Verarbeitung startet...')

                    # Process document synchronously in this context
                    try:
                        # Use asyncio.run to handle the async process_document call
                        result = asyncio.run(process_document(
                            temp_file_path, safe_filename, metadata,
                            on_phase=on_phase
                        ))
                    except Exception as e:
                        result = {"success": False, "error": str(e)}

                    # Final status update
                    if result["success"]:
                        update_table_row(uploaded_file.name, f'✅ {result["chunk_count"]} Textabschnitte', '✅ Hochgeladen')
                        st.session_state.processed_files.add(file_id)
                    else:
                        update_table_row(uploaded_file.name, f'Fehler: {result.get("error","Unbekannt")}', '❌ Error')

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
            st.session_state.just_uploaded = True
            
            # Clear the live update table to avoid duplicates
            table_placeholder.empty()
            
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
            
            # Schedule source update for next run
            st.session_state.need_source_update = True
            print(f"🔄 Nach Upload: {st.session_state.get('document_count', 0)} Dokumente, {st.session_state.get('knowledge_count', 0)} Notizen")
            
            # Show success message
            st.success(f"✅ Upload abgeschlossen! {successful_uploads} Datei(en) erfolgreich verarbeitet.")
            
            # Update header by triggering rerun only if files were successfully uploaded
            if successful_uploads > 0:
                # Set flag to prevent "already processed" message after rerun
                st.session_state.upload_just_completed = True
                st.rerun()

        elif already_processed_files and not new_files and not st.session_state.get("upload_just_completed", False):
            st.info("Alle Dateien wurden bereits verarbeitet")
    
    # Always display persistent upload status table if it exists
    if ("upload_status_table" in st.session_state and 
        st.session_state.upload_status_table and
        not st.session_state.get("currently_uploading", False)):
        st.subheader("📈 Upload-Status")
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


def render_document_management():
    """Rendert das Dokument-Management Interface"""
    st.markdown("### 🗑️ Dokument / Notiz löschen")
    
    # CSS for better readability
    st.markdown("""
    <style>
    .stTextArea textarea[disabled] {
        color: #000000 !important;
        background-color: #f8f9fa !important;
        opacity: 1 !important;
        font-family: 'Source Code Pro', monospace !important;
        line-height: 1.5 !important;
        border: 1px solid #dee2e6 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    div[data-testid="stTextArea"] textarea[disabled] {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }
    
    .stTextArea textarea {
        line-height: 1.5 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.session_state.sources:
        delete_filename = st.selectbox(
            "Dokument/Notiz selektieren", st.session_state.sources
        )

        # Preview
        st.markdown("### 📄 Vorschau")

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
                    # Note preview
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**Titel:** {metadata.get('title', 'Unbekannt')}")
                    with col2:
                        st.markdown(f"**Quelle:** {metadata.get('quelle', '–')}")
                    
                    st.text_area(
                        "Notizinhalt", 
                        content, 
                        height=400, 
                        disabled=True,
                        key=f"note_preview_{hash(delete_filename)}"
                    )
                else:
                    # Document preview
                    original_filename = metadata.get("original_filename", "")
                    st.markdown(f"**📄 Dokument:** {original_filename}")
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

            # Delete from database
            try:
                deleted_count = supabase_client.delete_documents_by_filename(delete_filename)
                st.code(f"🗨 SQL-Delete für '{delete_filename}' – {deleted_count} Einträge entfernt.")
                if deleted_count > 0:
                    db_deleted = True
            except Exception as e:
                st.error(f"Datenbank-Löschung fehlgeschlagen: {e}")

            # Delete from storage
            try:
                supabase_client.client.storage.from_("privatedocs").remove([delete_filename])
                storage_deleted = True
            except Exception as e:
                st.error(f"Löschen aus dem Speicher fehlgeschlagen: {e}")

            if storage_deleted and db_deleted:
                st.success("✅ Vollständig gelöscht.")
                st.session_state.need_source_update = True
                st.rerun()
            elif storage_deleted and not db_deleted:
                st.warning("⚠️ Dokument/Notiz im Storage gelöscht, aber kein Eintrag in der Datenbank gefunden.")
            elif not storage_deleted and db_deleted:
                st.warning("⚠️ Datenbankeinträge gelöscht, aber Dokument/Notiz im Storage konnte nicht entfernt werden.")
            else:
                st.error("❌ Weder Dokument/Notiz noch Datenbankeinträge konnten gelöscht werden.")

    else:
        st.info("Keine Dokumente/Notizen zur Löschung verfügbar.")


async def main():
    # Handle pending source updates
    if st.session_state.get("need_source_update", False):
        await update_available_sources()
        st.session_state.need_source_update = False
    
    # Handle pending chat processing
    if "pending_chat_input" in st.session_state:
        user_input = st.session_state.pending_chat_input
        del st.session_state.pending_chat_input
        await process_chat_input(user_input)
        st.rerun()
    
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

    # Render main menu
    render_main_menu()
    
    # Render submenu and content based on selection
    render_submenu_and_content()


def render_submenu_and_content():
    """Rendert Untermenü und Inhalte basierend auf der Auswahl (ohne URL-Wechsel)."""
    main_sel = st.session_state.get("selected_main_menu", "💬 Wunsch-Öle KI Assistent")

    if main_sel == "💬 Wunsch-Öle KI Assistent":
        sub_options = ["Chat", "Chat-Historie"]
        current_sub = st.session_state.get("selected_sub_menu", sub_options[0])
        st.markdown("<div class='submenu-scope'>", unsafe_allow_html=True)
        sel_sub = st.radio(
            "Untermenü Assistent",
            sub_options,
            index=sub_options.index(current_sub) if current_sub in sub_options else 0,
            horizontal=True,
            label_visibility="collapsed",
            key="assistant_submenu_text_radio",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        if sel_sub != current_sub:
            st.session_state.selected_sub_menu = sel_sub
            st.rerun()

        if sel_sub == "Chat-Historie":
            st.markdown("### 📜 Chat-Historie")
            st.info("🚧 Wird noch implementiert...")
        else:
            render_chat_interface()

    elif main_sel == "➕ Wissen hinzufügen":
        sub_options = ["Notiz hinzufügen", "Dokumente hochladen"]
        current_sub = st.session_state.get("selected_sub_menu", sub_options[0])
        st.markdown("<div class='submenu-scope'>", unsafe_allow_html=True)
        sel_sub = st.radio(
            "Untermenü Wissen",
            sub_options,
            index=sub_options.index(current_sub) if current_sub in sub_options else 0,
            horizontal=True,
            label_visibility="collapsed",
            key="knowledge_submenu_text_radio",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        if sel_sub != current_sub:
            st.session_state.selected_sub_menu = sel_sub
            st.rerun()

        if sel_sub == "Dokumente hochladen":
            render_document_upload()
        else:
            render_add_note()

    elif main_sel == "🗑️ Dokument anzeigen / löschen":
        render_document_management()


if __name__ == "__main__":
    asyncio.run(main())
