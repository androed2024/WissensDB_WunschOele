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

from supabase import create_client


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
import streamlit.components.v1 as st_components
import unicodedata
import re
import hashlib
import pandas as pd


def compute_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


from collections import defaultdict

# Logo-Pfad im Root-Verzeichnis
logo_path = "logo-wunschoele.png"

# Logo-Datei als base64 laden
with open(logo_path, "rb") as image_file:
    encoded = b64encode(image_file.read()).decode()

try:
    from st_aggrid import AgGrid, GridOptionsBuilder

    AGGRID = True
except Exception:
    AGGRID = False

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
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "5"))
DB_BATCH_SIZE = int(os.getenv("DB_BATCH_SIZE", "100"))

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]

# Auth-Client (für Login/Passwort ändern)
sb_auth = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Admin-Client (Service Role für Benutzerverwaltung - nur serverseitig verwenden!)
SB_ADMIN = create_client(SUPABASE_URL, os.environ["SUPABASE_SERVICE_ROLE_KEY"])


app_version = os.getenv("APP_VERSION", "0.0")
print("DEBUG VERSION:", os.getenv("APP_VERSION"))

from document_processing.ingestion import DocumentIngestionPipeline
from database.setup import SupabaseClient
from agent.agent import (
    RAGAgent,
    agent as rag_agent,
    format_source_reference,
    get_supabase_client,
    AgenticRetrievalOrchestrator,
    AgenticRetrievalConfig,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPartDelta,
)

st.set_page_config(
    page_title="Wunsch Öle Wissens-Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS für kompakteres Layout
st.markdown(
    """
<style>
/* 1) Alles Top-Padding/Margin global entfernen */
html, body { margin:0 !important; padding:0 !important; }
#root, [data-testid="stAppViewContainer"] { margin:0 !important; padding-top:0 !important; }

/* 2) Streamlit-Header, Toolbar, Deko und Status komplett entfernen (ohne reservierte Höhe) */
header[data-testid="stHeader"],
div[data-testid="stDecoration"],
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"] {
  display:none !important;
  height:0 !important;
  min-height:0 !important;
  visibility:hidden !important;
}

/* 3) Main-Container oben bündig machen */
section[data-testid="stMain"] { padding-top:0 !important; margin-top:0 !important; }

/* 4) Der eigentliche Block-Container: wirklich KEIN top spacing */
section[data-testid="stMain"] > div.block-container {
  padding-top:0 !important;
  margin-top:0 !important;
  padding-left:2rem !important;
  padding-right:2rem !important;
  max-width:none !important;
}

/* 5) Streamlit "Spacers" killen (verursachen oft den letzten Rest Abstand) */
div[data-testid="stSpacer"] {
  display:none !important;
  height:0 !important;
  min-height:0 !important;
  padding:0 !important;
  margin:0 !important;
}

/* 6) Ersten sichtbaren Block explizit bündig setzen */
section[data-testid="stMain"] > div.block-container > div:first-child {
  margin-top:0 !important;
  padding-top:0 !important;
}

/* 7) Tabs/Controls kompakter (optional) */
.stTabs [data-baseweb="tab-list"] { margin-top:0 !important; }
</style>
""",
    unsafe_allow_html=True,
)


def render_header():
    doc_count = st.session_state.get("document_count", 0)
    note_count = st.session_state.get("knowledge_count", 0)


    st.markdown(
        f"""
    <div style="height:4px;"></div>
    <div class="header-flex" style="display:flex;justify-content:space-between;align-items:center;margin:0;padding:0;">
        <div class="header-title-wrap" style="display:flex;align-items:center;">
            <img src="data:image/png;base64,{encoded}" alt="Logo" style="height:32px;margin-right:8px;">
            <span style="font-size:18px;font-weight:600;">Wunsch-Öle Wissens Agent</span>
            <span style="color:#007BFF;font-size:12px;margin-left:8px;">🔧 Version: {app_version}</span>
        </div>
        <div style="display:flex; gap:16px; align-items:center; font-size:12px;">
            <span>📄 Dokumente: {doc_count} &nbsp;&nbsp; 🧠 Notizen: {note_count}</span>
        </div>
    </div>
    <hr style="margin:2px 0 6px 0;border-top-width:1px;">
    """,
        unsafe_allow_html=True,
    )
    # ➜ direkt nach dem Markdown das Popover rendern:
    user_menu()


supabase_client = SupabaseClient()

# Agent instances will be created with appropriate client (Service for admin, User for others)
def get_agentic_orchestrator():
    """Get AgenticRetrievalOrchestrator with appropriate client for user role."""
    sb = get_db_client_for_admin_operations()
    return AgenticRetrievalOrchestrator(db_client=sb)

def get_rag_agent():
    """Get RAGAgent with appropriate client for user role."""
    sb = get_db_client_for_admin_operations()
    return RAGAgent(db_client=sb)

async def run_agentic_retrieval(user_input: str, config: AgenticRetrievalConfig) -> Dict[str, Any]:
    """
    Run agentic retrieval and return the result.
    
    Args:
        user_input: User's question
        config: Agentic retrieval configuration
        
    Returns:
        Dict with answer and source chunks
    """
    print(f"\n🔵 [USER INPUT] Original-Frage (Agentisch): {user_input}")
    
    try:
        orchestrator = get_agentic_orchestrator()
        result = await orchestrator.run(user_input, config)
        
        # Set last_match for UI compatibility (store in session_state for persistence)
        st.session_state['last_agentic_matches'] = result.get('source_chunks', [])
            
        return result
        
    except Exception as e:
        print(f"❌ Agentisches Retrieval Fehler: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": "Es ist ein Fehler beim agentischen Retrieval aufgetreten.",
            "source_chunks": [],
            "error": str(e)
        }

def _build_user_client(access_token: str):
    """Erzeuge einen Supabase-Client, der mit dem User-JWT gegen PostgREST authentifiziert ist (RLS greift)."""
    sb = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    sb.postgrest.auth(access_token)
    return sb

def ensure_login():
    if "user" in st.session_state and "sb_user" in st.session_state:
        return

    st.subheader("Anmelden")

    # Atomare Form: vermeidet Rerun-Rennen & Autofill-Probleme
    with st.form("login_form", clear_on_submit=False):
        st.text_input("E-Mail", key="login_email", value=st.session_state.get("login_email",""))
        st.text_input("Passwort", type="password", key="login_pw")
        submit = st.form_submit_button("Anmelden", use_container_width=True)

    if not submit:
        st.stop()

    # Werte NACH Submit aus session_state ziehen (stabil bei Autofill)
    email = (st.session_state.get("login_email") or "").strip()
    pw    = st.session_state.get("login_pw") or ""

    if not email or not pw:
        st.error("Bitte E-Mail und Passwort eingeben.")
        st.stop()

    try:
        # SDK-Kompatibilität (dict vs. named args)
        try:
            res = sb_auth.auth.sign_in_with_password({"email": email, "password": pw})
        except TypeError:
            res = sb_auth.auth.sign_in_with_password(email=email, password=pw)

        st.session_state["session"] = res.session
        st.session_state["user"]    = res.user
        st.session_state["roles"]   = (res.user.app_metadata or {}).get("roles", [])
        st.session_state["sb_user"] = _build_user_client(res.session.access_token)
        st.rerun()

    except Exception as e:
        st.error(f"Login fehlgeschlagen: {e}")
        st.stop()


def get_sb_user():
    """Hole den User-Client (mit JWT)."""
    return st.session_state.get("sb_user")

def get_db_client_for_admin_operations():
    """
    Hole den passenden DB-Client für Admin-Operationen.
    Admins bekommen Service-Client für uneingeschränkten Zugriff,
    normale User bekommen User-Client mit RLS.
    """
    if has_role("admin"):
        return SB_ADMIN  # Service-Client für Admins
    else:
        return get_sb_user()  # User-Client für normale User

def get_db_client_for_chat_operations():
    """
    Hole den passenden DB-Client für Chat-Historie-Operationen.
    Alle berechtigten User (chatbot_user, data_user, admin) bekommen Service-Client,
    da RLS-Policies für chat_history oft zu restriktiv sind.
    """
    if has_role("chatbot_user", "data_user", "admin"):
        return SB_ADMIN  # Service-Client für alle Chat-berechtigten User
    else:
        return get_sb_user()  # User-Client als Fallback

def get_db_client_for_upload_operations():
    """
    Hole den passenden DB-Client für Upload-/Document-Processing-Operationen.
    data_user und admin bekommen Service-Client für uneingeschränkten Upload-Zugriff,
    da RLS-Policies für document_metadata und rag_pages oft zu restriktiv sind.
    """
    if has_role("data_user", "admin"):
        return SB_ADMIN  # Service-Client für Upload-berechtigte User
    else:
        return get_sb_user()  # User-Client als Fallback

def has_role(*wanted):
    roles = st.session_state.get("roles", []) or []
    return ("admin" in roles) or any(r in roles for r in wanted)

def user_menu():
    """Popover rechts oben: Rollen anzeigen, Passwort ändern, Logout."""
    u = st.session_state.get("user")
    if not u: return
    roles = ", ".join(st.session_state.get("roles", [])) or "keine Rollen"

    with st.popover(f"User: {u.email}"):
        st.caption(f"Rollen: {roles}")
        old = st.text_input("Altes Passwort", type="password", key="pw_old")
        new = st.text_input("Neues Passwort", type="password", key="pw_new")
        if st.button("Passwort ändern"):
            try:
                # Re-Auth (falls nötig) & Passwort setzen
                sb_auth.auth.sign_in_with_password({"email": u.email, "password": old})
                sb_auth.auth.update_user({"password": new})
                st.success("Passwort geändert.")
            except Exception as e:
                st.error(f"Änderung fehlgeschlagen: {e}")
        if st.button("Logout"):
            sb_auth.auth.sign_out()
            for k in ("session","user","roles","sb_user"):
                st.session_state.pop(k, None)
            st.rerun()


# ==== Admin-Funktionen für Benutzerverwaltung ====

def list_users():
    """Liste aller Benutzer mit Rollen abrufen."""
    try:
        result = SB_ADMIN.auth.admin.list_users()
        users = []
        for user in result:
            roles = user.app_metadata.get("roles", []) if user.app_metadata else []
            users.append({
                "id": user.id,
                "email": user.email,
                "roles": roles
            })
        return users
    except Exception as e:
        st.error(f"Fehler beim Laden der Benutzer: {e}")
        return []

def set_roles(uid: str, roles: list):
    """Rollen für einen Benutzer setzen."""
    try:
        SB_ADMIN.auth.admin.update_user_by_id(uid, {"app_metadata": {"roles": roles}})
        return True
    except Exception as e:
        st.error(f"Fehler beim Setzen der Rollen: {e}")
        return False

def create_user(email: str, roles: list, password: str = None):
    """Neuen Benutzer erstellen."""
    try:
        user_data = {
            "email": email,
            "email_confirm": True,
            "app_metadata": {"roles": roles}
        }
        
        if password:
            user_data["password"] = password
            result = SB_ADMIN.auth.admin.create_user(user_data)
            st.success("Benutzer angelegt, kann sich sofort anmelden.")
        else:
            result = SB_ADMIN.auth.admin.create_user(user_data)
            # Recovery-Mail senden
            sb_auth.auth.reset_password_for_email(
                email, 
                options={"redirect_to": os.environ.get("APP_BASE_URL", "http://localhost:8501/")}
            )
            st.success("Benutzer angelegt. Recovery-Mail gesendet.")
        return True
    except Exception as e:
        st.error(f"Fehler beim Erstellen des Benutzers: {e}")
        return False

def delete_user(uid: str):
    """Benutzer löschen."""
    try:
        SB_ADMIN.auth.admin.delete_user(uid)
        return True
    except Exception as e:
        st.error(f"Fehler beim Löschen des Benutzers: {e}")
        return False

def send_recovery(email: str):
    """Recovery-Mail senden."""
    try:
        sb_auth.auth.reset_password_for_email(
            email, 
            options={"redirect_to": os.environ.get("APP_BASE_URL", "http://localhost:8501/")}
        )
        st.success("Recovery-Mail gesendet.")
        return True
    except Exception as e:
        st.error(f"Fehler beim Senden der Recovery-Mail: {e}")
        return False

# ==== Ende Admin-Funktionen ====


def render_agentic_config_sidebar():
    """Render sidebar with agentic retrieval configuration."""
    with st.sidebar:
        st.markdown("## 🤖 Agentisches Retrieval")
        
        # Toggle for agentic retrieval
        use_agentic = st.toggle(
            "Aktiviere Agentisches Retrieval",
            value=False,
            key="use_agentic_retrieval",
            help="Mehrstufige intelligente Suche mit Planung und Evidenz-Bündelung"
        )
        
        if use_agentic:
            st.markdown("### ⚙️ Konfiguration")
            
            # Max rounds
            max_rounds = st.slider(
                "Max. Such-Runden", 
                min_value=1, 
                max_value=6, 
                value=2, 
                key="agentic_max_rounds",
                help="Maximale Anzahl der Such-Runden (2 für schnelle Antworten, 4+ für komplexe Cross-Source-Fragen)"
            )
            
            # K per round
            k_per_round = st.slider(
                "Kandidaten pro Runde",
                min_value=5,
                max_value=30,
                value=15,
                key="agentic_k_per_round", 
                help="Anzahl der Dokumente die pro Such-Runde abgerufen werden"
            )
            
            # Token budget
            token_budget = st.slider(
                "Token-Budget",
                min_value=500,
                max_value=4000,
                value=2000,
                key="agentic_token_budget",
                help="Maximale Tokens für den Kontext (ca. 1500 Wörter)"
            )
            
            # Min similarity  
            min_similarity = st.slider(
                "Min. Ähnlichkeit",
                min_value=0.1,
                max_value=0.9,
                value=0.55,
                step=0.05,
                key="agentic_min_sim",
                help="Mindest-Ähnlichkeitsscore für Dokumente"
            )
            
            # Speed vs Quality toggle (prominent placement)
            quality_over_speed = st.toggle(
                "🎯 Qualität über Geschwindigkeit",
                value=False,
                key="agentic_quality_over_speed",
                help="Aktiviert: Vollständige Suche für beste Ergebnisse | Deaktiviert: Schnelle Antworten mit Early Stopping"
            )
            
            # Advanced options in expander
            with st.expander("🔧 Erweiterte Optionen"):
                recency_halflife = st.number_input(
                    "Recency Halflife (Tage)",
                    min_value=1,
                    max_value=365,
                    value=30,
                    key="agentic_recency_halflife",
                    help="Halbwertszeit für Aktualitäts-Bonus in Tagen"
                )
                
                doc_type_pref = st.selectbox(
                    "Dokumenttyp-Präferenz",
                    options=[None, "pdf", "txt", "manuell"],
                    index=0,
                    key="agentic_doc_type_pref",
                    help="Bevorzugter Dokumenttyp (None = alle)"
                )
                
                enable_filters = st.checkbox(
                    "Erweiterte Filter aktivieren",
                    value=True,
                    key="agentic_enable_filters",
                    help="Nutze Confidence-, Recency- und andere Filter"
                )
                
                early_stopping_threshold = st.slider(
                    "Early Stopping Threshold",
                    min_value=0.6,
                    max_value=0.9,
                    value=0.75,
                    step=0.05,
                    key="agentic_early_stopping",
                    help="Stoppe vorzeitig bei sehr guten Matches (höher = anspruchsvoller)"
                )
            
            return AgenticRetrievalConfig(
                max_rounds=max_rounds,
                k_per_round=k_per_round,
                token_budget=token_budget,
                min_similarity=min_similarity,
                recency_halflife_days=recency_halflife,
                doc_type_preference=doc_type_pref,
                enable_filters=enable_filters,
                early_stopping_threshold=early_stopping_threshold,
                quality_over_speed=quality_over_speed
            )
        
        return None


if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# Initialize chat history session state
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
    on_phase: Optional[PhaseCB] = None,
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
    # Use appropriate client (Service for data_user and admin, User for others)
    sb = get_db_client_for_upload_operations()
    pipeline = DocumentIngestionPipeline(db_client=sb)
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
            lambda: pipeline.process_file(
                file_path, metadata, on_phase=wrapped_on_phase
            ),
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
        try:
            # 'chunks' enthält APIResponse-Objekte; wir flatten auf Row-Dicts
            rows = []
            for resp in chunks:
                # Supabase-Python: resp.data ist i.d.R. eine Liste mit genau 1 Row (der Insert)
                if hasattr(resp, "data") and isinstance(resp.data, list):
                    rows.extend(resp.data)
                elif isinstance(resp, dict):
                    rows.append(resp)
                else:
                    # Unbekanntes Format -> skip
                    continue

            for i, row in enumerate(rows):
                emb = row.get("embedding") if isinstance(row, dict) else None
                txt = row.get("content", "") if isinstance(row, dict) else ""
                emb_len = len(emb) if isinstance(emb, (list, tuple)) else (emb.shape[0] if hasattr(emb, "shape") else 0)
                print(f"Chunk {i+1}: Embedding: {emb_len} Werte | Text: {txt[:100].replace(chr(10), ' ')}...")
        except Exception as e:
            print(f"⚠️ Embedding-Check übersprungen: {e}")

        
        
        # Finalisierung abgeschlossen
        if on_phase:
            on_phase("finalize", 1, 1)

        return {"success": True, "file_path": file_path, "chunk_count": len(chunks)}
    except Exception as e:
        import traceback

        print(f"Fehler bei der Bearbeitung des Dokuments: {str(e)}")
        print(traceback.format_exc())
        return {"success": False, "file_path": file_path, "error": str(e)}


async def run_agent_with_streaming(user_input: str, rag_agent_instance):
    # Log der Original-Frage vom User
    print(f"\n🔵 [USER INPUT] Original-Frage: {user_input}")

    async with rag_agent_instance.agent.iter(
        user_input,
        deps={"kb_search": rag_agent_instance.kb_search},
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


def get_chat_history(search_term: str = "") -> List[Dict]:
    """Holt Chat-Historie aus Supabase mit optionaler Wildcard-Suche"""
    try:
        # Service-Client für alle Chat-berechtigten User (chatbot_user, data_user, admin)
        sb = get_db_client_for_chat_operations()
        query = sb.table("chat_history").select("*")

        if search_term.strip():
            # Wildcard-Suche auf Frage-Feld
            search_pattern = f"%{search_term}%"
            query = query.ilike("question", search_pattern)

        response = query.order("created_at", desc=True).execute()
        return response.data or []
    except Exception as e:
        print(f"Fehler beim Abrufen der Chat-Historie: {e}")
        return []


def _chat_history_df(chat_history: List[Dict]) -> pd.DataFrame:
    """Hilfsfunktion für das DataFRame (chat_history tabelle)"""
    rows = []
    for ch in chat_history:
        q = (ch.get("question") or "").strip().replace("\n", " ")
        # sehr kompakt kürzen
        if len(q) > 120:
            q = q[:117] + "..."
        # Nur Datumsteil anzeigen (YYYY-MM-DD)
        ts = (ch.get("created_at") or "")[:10]
        rows.append(
            {
                "id": ch.get("id"),
                "Datum": ts,
                "Frage": q,
            }
        )
    df = pd.DataFrame(rows)
    # stabile Sortierung: neu → alt
    if "Datum" in df.columns:
        df = df.sort_values("Datum", ascending=False)
    # Index zurücksetzen und nicht als Spalte anzeigen
    df = df.reset_index(drop=True)
    return df

def render_chat_history():
    """Suche oben, klickbare Tabelle links (AgGrid) und Details rechts."""

    # ── Suche ──────────────────────────────────────────────────────────────────────
    search_term = st.text_input(
        "🔎 Suche in Fragen (Wildcard-Suche)",
        value="",
        placeholder="Geben Sie Suchbegriff ein...",
        key="chat_history_search_input",
        help="Die Suche erfolgt automatisch während der Eingabe"
    )

    # ── Daten ─────────────────────────────────────────────────────────────────────
    # Live-Suche: Verwende den aktuellen search_term direkt
    chat_history = get_chat_history(search_term.strip())
    if not chat_history:
        st.info(
            f"Keine Chats gefunden für: '{search_term.strip()}'"
            if search_term.strip()
            else "Keine Chat-Historie verfügbar."
        )
        return

    chat_df = _chat_history_df(chat_history)  # id, Datum, Frage
    grid_df = chat_df[["Datum", "Frage", "id"]].reset_index(drop=True)

    # Auswahl-Management: Reset bei Suche oder Initialisierung
    current_selected_id = st.session_state.get("selected_chat_id")
    available_ids = set(grid_df["id"].tolist()) if len(grid_df) > 0 else set()
    
    # Reset selection wenn aktuelle ID nicht in gefilterten Ergebnissen ist
    if current_selected_id not in available_ids and len(grid_df) > 0:
        st.session_state.selected_chat_id = grid_df.loc[0, "id"]
        print(f"[CHAT] reset selected_chat_id -> {st.session_state.selected_chat_id} (search filter changed)")
    elif not current_selected_id and len(grid_df) > 0:
        st.session_state.selected_chat_id = grid_df.loc[0, "id"]
        print(f"[CHAT] init selected_chat_id -> {st.session_state.selected_chat_id}")

    # ── Layout ────────────────────────────────────────────────────────────────────
    col_list, col_detail = st.columns([1, 2])

    # ── Hilfsfunktion: Auswahl sicher aus AgGridReturn holen ──────────────────────
    def _get_selected_rows(grid_return) -> list[dict]:
        try:
            # Dict-Return (ältere Versionen)
            if isinstance(grid_return, dict):
                rows = grid_return.get("selected_rows") or grid_return.get("selectedRows") or []
            else:
                # Neuere Version: Attribute können DataFrames sein
                rows = getattr(grid_return, "selected_rows", None)
                if rows is None:
                    rows = getattr(grid_return, "selectedRows", None)

            # Normalisieren
            if rows is None:
                return []
            import pandas as _pd
            if isinstance(rows, _pd.DataFrame):
                return rows.to_dict("records")
            if isinstance(rows, list):
                return rows
            # Fallback: unbekannter Typ → in Liste packen, wenn Mapping
            try:
                return [dict(rows)]
            except Exception:
                return []
        except Exception as e:
            print(f"[CHAT] _get_selected_rows error: {e}")
            return []

    # ── Linke Spalte: Tabelle / AgGrid ────────────────────────────────────────────
    with col_list:
        st.markdown("**Chat-Liste:**")

        if AGGRID:
            from st_aggrid import GridOptionsBuilder
            gb = GridOptionsBuilder.from_dataframe(grid_df)
            gb.configure_column("id", hide=True)
            gb.configure_column("Datum", headerName="Datum", width=110, resizable=True)
            gb.configure_column("Frage", headerName="Frage", flex=1, wrapText=True, autoHeight=True)
            gb.configure_default_column(resizable=True, sortable=True, filter=True)
            gb.configure_selection(selection_mode="single", use_checkbox=False)
            gb.configure_grid_options(
                rowHeight=28,
                suppressRowClickSelection=False,
                animateRows=False,
                domLayout="autoHeight",
                rememberSelection=True,
            )

            grid_key = f"chat_history_grid_v5_{len(grid_df)}_{search_term.strip()}"

            # Neu: update_on (neue API)  → Fallback: update_mode (alte API)
            try:
                grid = AgGrid(
                    grid_df,
                    gridOptions=gb.build(),
                    update_on=["selectionChanged"],     # <- wichtig
                    fit_columns_on_grid_load=True,
                    allow_unsafe_jscode=True,
                    enable_enterprise_modules=False,
                    theme="alpine",
                    key=grid_key,
                )
            except TypeError:
                from st_aggrid import GridUpdateMode
                grid = AgGrid(
                    grid_df,
                    gridOptions=gb.build(),
                    update_mode=GridUpdateMode.SELECTION_CHANGED,
                    fit_columns_on_grid_load=True,
                    allow_unsafe_jscode=True,
                    enable_enterprise_modules=False,
                    theme="alpine",
                    key=grid_key,
                )

            print(f"[CHAT] grid type: {type(grid)}")
            selected_rows = _get_selected_rows(grid)
            print(f"[CHAT] selected_rows len = {len(selected_rows)}")

            if selected_rows:
                row0 = selected_rows[0]
                print(f"[CHAT] first selected row keys: {list(row0.keys())}")
                sel_id = row0.get("id")

                # Falls 'id' durch Hide nicht mitkommt → anhand Datum/Frage rekonstruieren
                if not sel_id:
                    try:
                        match = grid_df[
                            (grid_df["Datum"] == row0.get("Datum")) &
                            (grid_df["Frage"] == row0.get("Frage"))
                        ]
                        if len(match) == 1:
                            sel_id = match.iloc[0]["id"]
                            print(f"[CHAT] reconstructed id -> {sel_id}")
                    except Exception as e:
                        print(f"[CHAT] reconstruct failed: {e}")

                if sel_id and sel_id != st.session_state.get("selected_chat_id"):
                    st.session_state.selected_chat_id = sel_id
                    print(f"[CHAT] >> selection changed -> {sel_id} (rerun)")
                    st.rerun()
                else:
                    print("[CHAT] selection unchanged or id missing")

            # Kompakteres Grid CSS
            st.markdown(
                """
                <style>
                .ag-theme-alpine .ag-cell { padding: 2px 4px !important; font-size: 12px !important; line-height: 1.2 !important; }
                .ag-theme-alpine .ag-header-cell-label { padding: 2px 4px !important; font-size: 12px !important; font-weight: 600 !important; }
                .ag-theme-alpine .ag-row { min-height: 28px !important; }
                </style>
                """,
                unsafe_allow_html=True,
            )
        else:
            # Fallback ohne AgGrid
            options = grid_df["id"].tolist()
            labels = [f"{row['Datum']} — {row['Frage']}" for _, row in grid_df.iterrows()]
            current = st.session_state.get("selected_chat_id")
            idx = options.index(current) if current in options else 0 if options else None
            if options:
                selection = st.radio(" ", options=options, index=idx,
                                     format_func=lambda x: labels[options.index(x)])
                if selection and selection != current:
                    st.session_state.selected_chat_id = selection
                    print(f"[CHAT] radio -> selection changed to {selection} (rerun)")
                    st.rerun()
            else:
                st.info("Keine Einträge.")

    # ── Rechte Spalte: Chat-Details ───────────────────────────────────────────────
    with col_detail:
        st.markdown("**Chat-Details:**")
        sel_id = st.session_state.get("selected_chat_id")
        print(f"[CHAT] details sel_id = {sel_id}")

        if sel_id:
            selected_chat = next((c for c in chat_history if c.get("id") == sel_id), None)
            print(f"[CHAT] found selected_chat? -> {bool(selected_chat)}")
            if selected_chat:
                with st.chat_message("user"):
                    st.markdown(selected_chat.get("question", ""))
                with st.chat_message("assistant"):
                    st.markdown(selected_chat.get("answer", ""))
            else:
                st.info("Chat nicht gefunden.")
        else:
            st.info("Wähle links einen Eintrag aus, um Frage & Antwort zu sehen.")



async def update_available_sources():
    try:
        # response = (supabase_client.client.table("rag_pages").select("url, metadata").execute()

        # neu (Service-Client für data_user und admin, RLS für andere):
        sb = get_db_client_for_upload_operations()
        response = sb.table("rag_pages").select("url, metadata").execute()


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
    # 0) Login erzwingen
    ensure_login()

    # Erst Daten laden, dann Header rendern
    await update_available_sources()
    render_header()
    
    # Render agentic config sidebar and get config
    agentic_config = render_agentic_config_sidebar()

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

    # ---- Rollenbasierte Tabs ----
    tab_labels = []
    
    # Chat-Bereich (chatbot_user, data_user, admin)
    if has_role("chatbot_user", "data_user", "admin"):
        tab_labels.append("💬 Wunsch-Öle KI Assistent")
    
    # Wissen hinzufügen / Doku hochladen (data_user, admin)  
    if has_role("data_user", "admin"):
        tab_labels.append("➕ Wissen hinzufügen")
    
    # Dokumente anzeigen / löschen (data_user, admin)
    if has_role("data_user", "admin"):
        tab_labels.append("🗑️ Dokument anzeigen / löschen")
    
    # Benutzerverwaltung (nur admin)
    if has_role("admin"):
        tab_labels.append("👤 Benutzerverwaltung")
    
    if not tab_labels:
        st.warning("Keine berechtigten Bereiche mit deinen Rollen.")
        return
        
    tabs = st.tabs(tab_labels)
    current_tab = 0

    def render_user_management():
        """Renders the user management UI for admins."""
        st.markdown("### 👤 Benutzerverwaltung")
        st.markdown("---")
        
        # Neuen Benutzer anlegen
        st.markdown("#### ➕ Neuen Benutzer anlegen")
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                new_email = st.text_input("E-Mail-Adresse", key="new_user_email")
                new_password = st.text_input("Passwort (optional - wenn leer, wird Recovery-Mail gesendet)", 
                                           type="password", key="new_user_password")
            
            with col2:
                st.markdown("**Rollen:**")
                role_admin = st.checkbox("admin", key="new_user_admin")
                role_data = st.checkbox("data_user", key="new_user_data")
                role_chat = st.checkbox("chatbot_user", key="new_user_chat")
            
            if st.button("Benutzer anlegen", type="primary"):
                if new_email.strip():
                    roles = []
                    if role_admin: roles.append("admin")
                    if role_data: roles.append("data_user") 
                    if role_chat: roles.append("chatbot_user")
                    
                    if roles:
                        password = new_password.strip() if new_password.strip() else None
                        if create_user(new_email.strip(), roles, password):
                            st.rerun()
                    else:
                        st.error("Mindestens eine Rolle muss ausgewählt werden.")
                else:
                    st.error("E-Mail-Adresse ist erforderlich.")
        
        st.markdown("---")
        
        # Bestehende Benutzer verwalten
        st.markdown("#### 👥 Bestehende Benutzer verwalten")
        
        users = list_users()
        if users:
            for user in users:
                with st.container():
                    col1, col2, col3 = st.columns([2, 3, 2])
                    
                    with col1:
                        st.markdown(f"**{user['email']}**")
                        st.caption(f"ID: {user['id'][:8]}...")
                    
                    with col2:
                        st.markdown("**Rollen:**")
                        user_roles = user.get('roles', [])
                        
                        # Checkboxen für Rollen (unique keys mit user_id)
                        admin_checked = st.checkbox("admin", 
                                                  value="admin" in user_roles,
                                                  key=f"admin_{user['id']}")
                        data_checked = st.checkbox("data_user", 
                                                 value="data_user" in user_roles,
                                                 key=f"data_{user['id']}")
                        chat_checked = st.checkbox("chatbot_user", 
                                                 value="chatbot_user" in user_roles,
                                                 key=f"chat_{user['id']}")
                    
                    with col3:
                        # Action Buttons
                        if st.button("💾 Rollen speichern", key=f"save_{user['id']}"):
                            new_roles = []
                            if admin_checked: new_roles.append("admin")
                            if data_checked: new_roles.append("data_user")
                            if chat_checked: new_roles.append("chatbot_user")
                            
                            if set_roles(user['id'], new_roles):
                                st.success("Rollen aktualisiert!")
                                st.rerun()
                        
                        if st.button("📧 Recovery senden", key=f"recovery_{user['id']}"):
                            if send_recovery(user['email']):
                                st.rerun()
                        
                        # Sicherheitsabfrage für Löschen
                        delete_confirm = st.checkbox("Bestätigen", key=f"confirm_{user['id']}")
                        if st.button("🗑️ Löschen", 
                                   disabled=not delete_confirm, 
                                   key=f"delete_{user['id']}"):
                            if delete_user(user['id']):
                                st.success("Benutzer gelöscht!")
                                st.rerun()
                    
                    st.markdown("---")
        else:
            st.info("Keine Benutzer gefunden oder Fehler beim Laden.")
        
        # Hinweise
        st.markdown("#### ℹ️ Hinweise")
        st.info("""
        - Nach Rollen-Updates müssen betroffene Benutzer sich neu einloggen
        - Der Service-Client wird nur serverseitig verwendet 
        - Alle RLS-Policies bleiben für normale Datenbankabfragen aktiv
        """)

    async def render_delete_preview_ui():
        """Renders the complete document/note preview and delete UI."""
        # Custom CSS for better text readability in preview areas
        st.markdown(
        """
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
        """,
            unsafe_allow_html=True,
        )

        # Zeige Vorschau nur wenn Benutzer die del-Berechtigung hat
        if st.session_state.sources and has_role("data_user", "admin"):
            delete_filename = st.selectbox(
                "Dokument/Notiz selektieren", st.session_state.sources
            )

            # Vorschau anzeigen mit Button auf gleicher Höhe
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    "<h5 style='margin-top: 0rem; margin-bottom: 0.5rem;'>📄 Vorschau</h5>",
                    unsafe_allow_html=True,
                )
            with col2:
                delete_button_pressed = st.button(
                    "🗑️ Ausgewähltes Dokument/Notiz löschen", key="delete_doc_button"
                )

            # Metadaten und Inhalt aus Supabase holen
            try:
                sb = get_db_client_for_admin_operations()
                res = (
                    sb.table("rag_pages")
                    .select("content", "metadata", "chunk_number")
                    .eq("url", delete_filename)
                    .order("chunk_number")  # Chunks in richtiger Reihenfolge
                    .execute()
                )

                if res.data:
                    # Für manuelle Notizen alle Chunks zusammensetzen
                    first_entry = res.data[0]
                    metadata = first_entry.get("metadata", {})
                    source = metadata.get("source", "")
                    
                    if source == "manuell" and len(res.data) > 1:
                        # Mehrere Chunks -> zusammensetzen (agentisch gechunkte Notiz)
                        content_parts = []
                        for entry in res.data:
                            chunk_content = entry.get("content", "")
                            if chunk_content.strip():
                                content_parts.append(chunk_content.strip())
                        
                        # Chunks mit Zeilentrennung zusammenfügen
                        content = "\n\n".join(content_parts)
                        st.info(f"📄 Lange Notiz rekonstruiert aus {len(res.data)} agentisch erzeugten Chunks")
                    else:
                        # Einzelner Chunk oder andere Dokumente
                        content = first_entry.get("content", "")

                    if source == "manuell":
                        # Put title and source side by side
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(
                                f"**Titel:** {metadata.get('title', 'Unbekannt')}"
                            )
                        with col2:
                            st.markdown(f"**Quelle:** {metadata.get('quelle', '–')}")

                        # Use text_area with proper line breaks and scrolling like text files
                        st.text_area(
                            "Notizinhalt",
                            content,
                            height=400,
                            disabled=True,
                            key=f"note_preview_{hash(delete_filename)}",
                        )
                    else:
                        # Unterscheidung zwischen PDF und TXT Dateien
                        original_filename = metadata.get("original_filename", "")
                        file_extension = metadata.get("file_extension", "").lower()

                        # Fallback: Determine file extension from filename if not in metadata
                        if not file_extension and original_filename:
                            if "." in original_filename:
                                file_extension = (
                                    "." + original_filename.lower().split(".")[-1]
                                )
                            else:
                                file_extension = ""

                        if file_extension == ".pdf":
                            # Original PDF anzeigen
                            try:
                                st.markdown("**📄 Original-PDF Vorschau:**")

                                # Create signed URL dynamically for PDF preview
                                client = get_supabase_client()
                                try:
                                    res = client.storage.from_(
                                        "privatedocs"
                                    ).create_signed_url(original_filename, 3600)
                                    signed_url = res.get("signedURL")

                                    if signed_url and signed_url != "#":
                                        st.markdown("**PDF-Inhalt:**")

                                        # Try multiple PDF viewer approaches for maximum compatibility

                                        # Approach 1: Try Mozilla PDF.js viewer first (most reliable)
                                        pdfjs_url = f"https://mozilla.github.io/pdf.js/web/viewer.html?file={signed_url}"

                                        st_components.html(
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
                                        st.warning(
                                            "⚠️ PDF-Vorschau konnte nicht geladen werden. Signed URL nicht verfügbar."
                                        )
                                        st.info(f"Debug: Response: {res}")

                                except Exception as url_error:
                                    st.error(
                                        f"❌ Fehler beim Erstellen der PDF-Vorschau URL: {url_error}"
                                    )
                                    st.info("Versuche alternative Anzeige...")

                                    # Fallback: Show text content if available
                                    if content and content.strip():
                                        st.markdown("**Extrahierter Text-Inhalt:**")
                                        st.text_area(
                                            "PDF Textinhalt",
                                            content,
                                            height=400,
                                            disabled=True,
                                            key=f"pdf_text_preview_{hash(delete_filename)}",
                                        )
                                    else:
                                        st.warning(
                                            "Keine PDF-Vorschau oder Textinhalt verfügbar."
                                        )

                            except Exception as e:
                                st.error(f"❌ Fehler beim Laden der PDF-Vorschau: {e}")
                                # Show text content as fallback
                                if content and content.strip():
                                    st.markdown(
                                        "**Extrahierter Text-Inhalt (Fallback):**"
                                    )
                                    st.text_area(
                                        "PDF Textinhalt",
                                        content,
                                        height=400,
                                        disabled=True,
                                        key=f"pdf_fallback_preview_{hash(delete_filename)}",
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
                                    st.markdown(
                                        f"**Dateigröße:** {metadata.get('file_size_bytes', 0)} Bytes"
                                    )
                                with col3:
                                    st.markdown(
                                        f"**Chunks:** {metadata.get('chunk_count', 1)}"
                                    )

                                st.text_area(
                                    "Dateiinhalt",
                                    content,
                                    height=400,
                                    disabled=True,
                                    key=f"txt_preview_{hash(delete_filename)}",
                                )
                            except Exception as e:
                                st.error(f"Fehler beim Laden der TXT-Vorschau: {e}")
                        else:
                            # Fallback für andere Dateitypen
                            st.markdown(
                                f"**📄 Dokument Vorschau ({original_filename}):**"
                            )
                            st.markdown(
                                f"**Dateityp:** {file_extension if file_extension else 'Unbekannt'}"
                            )
                            st.markdown("**Inhalt:**")
                            st.text_area(
                                "Dokumentinhalt",
                                content,
                                height=400,
                                disabled=True,
                                key=f"doc_preview_{hash(delete_filename)}",
                            )
                else:
                    st.info("Keine Vorschau verfügbar.")

            except Exception as e:
                st.error(f"Fehler beim Laden der Vorschau: {e}")

            if delete_button_pressed:
                st.write("Dateiname zur Löschung:", delete_filename)

                storage_deleted = db_deleted = False

                # Erst Datenbank, dann Storage löschen (umgekehrte Reihenfolge)
                try:
                    print(f"🗑️ Lösche Datenbankeinträge für: {delete_filename}")
                    
                    # Use appropriate client (Service for admin, User for others)
                    sb = get_db_client_for_admin_operations()
                    delete_result = sb.table("rag_pages").delete().eq("url", delete_filename).execute()
                    deleted_count = len(delete_result.data or [])
                    
                    st.code(
                        f"🩨 SQL-Delete für '{delete_filename}' – {deleted_count} Einträge entfernt."
                    )
                    if deleted_count > 0:
                        db_deleted = True
                        print(
                            f"✅ {deleted_count} Datenbankeinträge erfolgreich gelöscht"
                        )
                    else:
                        print(
                            f"⚠️ Keine Datenbankeinträge für {delete_filename} gefunden"
                        )
                except Exception as e:
                    error_msg = str(e).lower()
                    if "403" in error_msg or "forbidden" in error_msg or "not allowed" in error_msg:
                        st.error("❌ Löschung nicht erlaubt. Nur Administratoren können Dokumente löschen.")
                        print(f"🚫 RLS-Blockierung: Benutzer hat keine Löschberechtigung für {delete_filename}")
                    else:
                        st.error(f"Datenbank-Löschung fehlgeschlagen: {e}")
                        print(f"❌ Datenbank-Löschung fehlgeschlagen: {e}")
                    db_deleted = False

                try:
                    print(f"🗑️ Lösche Storage-Datei: {delete_filename}")
                    # Use appropriate client (Service for admin, User for others)
                    sb = get_db_client_for_admin_operations()
                    sb.storage.from_("privatedocs").remove(
                        [delete_filename]
                    )
                    storage_deleted = True
                    print(f"✅ Storage-Datei erfolgreich gelöscht")
                except Exception as e:
                    error_msg = str(e).lower()
                    if "403" in error_msg or "forbidden" in error_msg or "not allowed" in error_msg:
                        st.error("❌ Storage-Löschung nicht erlaubt. Nur Administratoren können Dateien aus dem Speicher löschen.")
                        print(f"🚫 RLS-Blockierung: Benutzer hat keine Storage-Löschberechtigung für {delete_filename}")
                    else:
                        st.error(f"Löschen aus dem Speicher fehlgeschlagen: {e}")
                        print(f"❌ Storage-Löschung fehlgeschlagen: {e}")

                # Zusätzliche Verifikation: Prüfe ob wirklich gelöscht
                try:
                    print(
                        f"🔍 Verifikation: Prüfe ob {delete_filename} wirklich gelöscht wurde..."
                    )
                    sb = get_db_client_for_admin_operations()
                    verify_result = (
                        sb.table("rag_pages")
                        .select("id,url")
                        .eq("url", delete_filename)
                        .execute()
                    )
                    remaining_entries = len(verify_result.data or [])
                    if remaining_entries > 0:
                        print(
                            f"⚠️ WARNUNG: {remaining_entries} Einträge für {delete_filename} sind noch in der Datenbank!"
                        )
                        st.warning(
                            f"⚠️ {remaining_entries} Einträge sind noch in der Datenbank vorhanden!"
                        )
                        # Versuche nochmal zu löschen
                        print("🔄 Versuche erneute Löschung...")
                        retry_deleted = supabase_client.delete_documents_by_filename(
                            delete_filename
                        )
                        print(
                            f"🔄 Zweiter Löschversuch: {retry_deleted} Einträge entfernt"
                        )
                    else:
                        print(
                            f"✅ Verifikation erfolgreich: Keine Einträge für {delete_filename} gefunden"
                        )
                except Exception as e:
                    print(f"❌ Verifikation fehlgeschlagen: {e}")
                    st.error(f"Verifikation fehlgeschlagen: {e}")

                if storage_deleted and db_deleted:
                    st.success("✅ Vollständig gelöscht.")
                    # Remove from processed files list so it can be re-uploaded
                    if "processed_files" in st.session_state:
                        # Remove all entries that match this filename
                        files_to_remove = [
                            f
                            for f in st.session_state.processed_files
                            if delete_filename in f
                        ]
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

    # Chat-Tab (immer der erste verfügbare Tab für berechtigte Benutzer)
    if has_role("chatbot_user", "data_user", "admin"):
        with tabs[current_tab]:
            # Erstelle Untermenü mit zwei Optionen (ohne doppelten Titel)
            chat_tab1, chat_tab2 = st.tabs(["Chat", "Chat Historie"])

        with chat_tab1:
            # Vereinfachtes Layout ohne feste Header
            st.markdown(
                """
            <style>
            /* Basis-Styling ohne feste Positionierung */
            .stChatInput textarea {
                min-height: 50px !important;
                border-radius: 8px !important;
                border: 2px solid #dee2e6 !important;
                padding: 8px 12px !important;
            }
            
            .stChatInput textarea:focus {
                border-color: #007BFF !important;
                box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
            }
            
            .chat-section {
                margin-top: 0.5rem;
            }
            
            /* Kompaktere Chat-Nachrichten */
            .stChatMessage {
                margin-bottom: 0.5rem !important;
            }
            
            /* Kompaktere Untertabs */
            .stTabs [data-baseweb="tab-list"] button {
                padding: 4px 10px !important;
                margin-right: 4px !important;
            }
            </style>
            
            """,
                unsafe_allow_html=True,
            )

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

                    # Choose between agentic and classic retrieval
                    rag_agent = None  # Initialize for source processing
                    
                    if agentic_config is not None:
                        # Use agentic retrieval (silently in background)
                        agentic_result = await run_agentic_retrieval(user_input, agentic_config)
                        full_response = agentic_result.get("answer", "")
                        message_placeholder.markdown(full_response)
                        
                    else:
                        # Use classic streaming retrieval
                        rag_agent = get_rag_agent()  # Only needed for classic retrieval
                        
                        # Reset akkumulierte Treffer für neue Frage (aber nicht None setzen!)
                        if hasattr(rag_agent, "last_match"):
                            rag_agent.last_match = []
                            print(f"🔄 Reset: Treffer-Akkumulator für neue Frage geleert")

                        async for chunk in run_agent_with_streaming(user_input, rag_agent):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")

                    # Chatbot Interface - Quellenangaben verarbeiten
                    pdf_sources = defaultdict(set)
                    txt_sources = defaultdict(set)
                    note_sources = defaultdict(set)

                    # Check for sources from both classic and agentic retrieval
                    source_chunks = []
                    
                    # Classic retrieval sources
                    if hasattr(rag_agent, "last_match") and rag_agent.last_match:
                        source_chunks = rag_agent.last_match
                    # Agentic retrieval sources
                    elif agentic_config is not None and 'last_agentic_matches' in st.session_state:
                        source_chunks = st.session_state['last_agentic_matches']
                    
                    if source_chunks:
                        print("--- Treffer im Retrieval ---")
                        # Score-basierte Filterung für Anzeige - identisch mit Retrieval-Schwelle
                        DISPLAY_MIN_SIM = float(os.getenv("RAG_MIN_SIM", "0.55"))
                        print(
                            f"🎯 Anzeige-Schwellenwert: {DISPLAY_MIN_SIM} (identisch mit Retrieval)"
                        )

                        # Sammle alle Matches und finde den besten Score
                        all_matches = []
                        best_score = 0

                        for match in source_chunks:
                            sim = match.get("similarity", 0)
                            if sim >= DISPLAY_MIN_SIM:
                                all_matches.append((match, sim))
                                best_score = max(best_score, sim)

                        # Intelligente Filterung: Weniger aggressiv für Cross-Source-Queries
                        # Check if we have potential cross-source results (different doc types/sources)
                        unique_sources = set()
                        metadata_matches = 0
                        
                        for match in source_chunks:
                            meta = match.get("metadata", {})
                            source_type = meta.get("source", "")
                            unique_sources.add(source_type)
                            if match.get("source_type") == "metadata_match":
                                metadata_matches += 1
                        
                        has_cross_source = len(unique_sources) > 1 or metadata_matches > 0
                        
                        if has_cross_source:
                            # Less aggressive filtering for cross-source queries
                            RELEVANCE_GAP_THRESHOLD = 0.20  # More lenient: 20% gap allowed
                            print("🔄 Cross-Source detected: Using lenient filtering")
                        else:
                            # Standard filtering for single-source queries
                            RELEVANCE_GAP_THRESHOLD = 0.13  # Standard: 13% gap

                        if best_score > 0:
                            gap_threshold = best_score - RELEVANCE_GAP_THRESHOLD
                            effective_threshold = max(DISPLAY_MIN_SIM, gap_threshold)
                            print(
                                f"🎯 Intelligente Filterung: Bester Score {best_score:.3f}, Effektiver Threshold: {effective_threshold:.3f} (Cross-Source: {has_cross_source})"
                            )
                        else:
                            effective_threshold = DISPLAY_MIN_SIM

                        for match, sim in all_matches:
                            # Verwende intelligenten Threshold
                            if sim < effective_threshold:
                                print(
                                    f"⚠️ Treffer gefiltert (Score {sim:.3f} < {effective_threshold:.3f} - zu weit vom besten Score entfernt)"
                                )
                                continue

                            meta = match.get("metadata", {})
                            # Versuche zuerst original_filename, dann url als Fallback
                            fn = meta.get("original_filename") or match.get("url", "")
                            pg = meta.get("page", 1)
                            source_type = meta.get("source", "")

                            if fn:
                                if source_type == "manuell":
                                    note_sources[fn].add(pg)
                                    print("✅ Notiz:", fn, "| Score:", sim)
                                else:
                                    # Kategorisiere basierend auf Dateierweiterung
                                    file_extension = (
                                        fn.lower().split(".")[-1] if "." in fn else ""
                                    )
                                    if file_extension == "txt":
                                        txt_sources[fn].add(pg)
                                        print("✅ TXT-Dokument:", fn, "| Score:", sim)
                                    else:
                                        pdf_sources[fn].add(pg)
                                        print(
                                            "✅ PDF-Dokument:",
                                            fn,
                                            "| Seite:",
                                            pg,
                                            "| Score:",
                                            sim,
                                        )

                    # Bereinige die Agent-Antwort von bestehenden Quellenangaben
                    import re

                    # ULTRA-aggressive Bereinigung: Entferne alles ab dem ersten "Quelle"
                    lines = full_response.split("\n")
                    cleaned_lines = []

                    for line in lines:
                        line_lower = line.lower().strip()
                        # Stoppe bei jeder Zeile, die "quelle" enthält
                        if (
                            "quelle" in line_lower
                            or "pdf öffnen" in line_lower
                            or ".pdf" in line_lower
                        ):
                            break
                        cleaned_lines.append(line)

                    full_response = "\n".join(cleaned_lines).strip()

                    # Entferne leere Zeilen am Ende
                    while full_response.endswith("\n\n\n"):
                        full_response = full_response[:-1]

                    # Prüfe ob Agent "keine Informationen" geantwortet hat (nur bei komplett leeren Antworten)
                    no_info_complete_phrases = [
                        "es liegen keine informationen zu dieser frage in der wissensdatenbank vor",
                        "ich habe keine daten zu diesem thema gefunden",
                        "diese information ist nicht in der wissensdatenbank verfügbar",
                    ]

                    response_lower = full_response.lower().strip()
                    # Nur unterdrücken wenn die Antwort HAUPTSÄCHLICH "keine Informationen" ist
                    has_no_info_response = any(
                        phrase in response_lower for phrase in no_info_complete_phrases
                    )

                    # Zusätzlich: Wenn Antwort sehr kurz ist UND "keine Informationen" enthält
                    if len(response_lower) < 200:  # Kurze Antworten
                        short_no_info_phrases = [
                            "keine informationen zu dieser frage",
                            "liegen keine informationen vor",
                        ]
                        has_no_info_response = has_no_info_response or any(
                            phrase in response_lower for phrase in short_no_info_phrases
                        )

                    if has_no_info_response:
                        print(
                            "🚫 Agent hat 'keine Informationen' geantwortet - unterdrücke Quellenanzeige"
                        )

                    # Füge saubere Quellenangaben hinzu
                    retrieval_type = "Agentisch" if agentic_config is not None else "Klassisch"
                    print(f"🔍 Debug ({retrieval_type}): PDF-Quellen gefunden: {dict(pdf_sources)}")
                    print(f"🔍 Debug ({retrieval_type}): TXT-Quellen gefunden: {dict(txt_sources)}")
                    print(f"🔍 Debug ({retrieval_type}): Notiz-Quellen gefunden: {dict(note_sources)}")
                    print(f"🔍 Debug ({retrieval_type}): Source chunks: {len(source_chunks) if source_chunks else 0}")
                    print(
                        f"🔍 Debug: Hat 'keine Informationen' Antwort: {has_no_info_response}"
                    )

                    # Nur Quellen anzeigen wenn Agent nicht "keine Informationen" geantwortet hat
                    if (
                        pdf_sources or txt_sources or note_sources
                    ) and not has_no_info_response:
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

                                if source_chunks:
                                    for match in source_chunks:
                                        meta = match.get("metadata", {})
                                        if meta.get("source") == "manuell" and (
                                            fn in meta.get("original_filename", "")
                                            or fn == meta.get("title", "")
                                        ):
                                            note_title = meta.get("title", fn)

                                            # Bestimme den korrekten Dateinamen für Storage-Link
                                            original_filename = meta.get(
                                                "original_filename", fn
                                            )
                                            storage_filename = meta.get(
                                                "storage_filename", original_filename
                                            )

                                            # Prüfe, ob Storage-Datei verfügbar ist
                                            has_storage = meta.get(
                                                "has_storage_file", True
                                            )  # Default True für Rückwärtskompatibilität
                                            source_filter = meta.get(
                                                "source_filter", "privatedocs"
                                            )

                                            if (
                                                has_storage
                                                and source_filter == "privatedocs"
                                            ):
                                                # Notiz mit Storage-Datei
                                                note_filename = (
                                                    storage_filename
                                                    or original_filename
                                                )
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
                                        print(
                                            f"🔍 Debug: Versuche signed URL für {note_filename} in bucket {bucket}"
                                        )

                                        # Erstelle signed URL für Notiz
                                        res = client.storage.from_(
                                            bucket
                                        ).create_signed_url(note_filename, 3600)
                                        signed_url = res.get("signedURL", "#")

                                        print(f"🔍 Debug: Signed URL Response: {res}")
                                        print(f"🔍 Debug: Signed URL: {signed_url}")

                                        # Signed URL ist bereit für die Verwendung

                                        if signed_url and signed_url != "#":
                                            print(
                                                f"✅ Signed URL für Notiz erstellt: {signed_url[:50]}..."
                                            )
                                            full_response += (
                                                f"\n**📝 Notiz:** {note_title}\n"
                                            )
                                            full_response += (
                                                f"[📄 Notiz öffnen]({signed_url})\n"
                                            )
                                        else:
                                            print(
                                                f"⚠️ Keine gültige Signed URL erhalten: {res}"
                                            )
                                            full_response += (
                                                f"\n**📝 Notiz:** {note_title}\n"
                                            )
                                            full_response += f"(Link nicht verfügbar)\n"

                                    except Exception as e:
                                        print(
                                            f"⚠️ Signed URL für Notiz fehlgeschlagen: {e}"
                                        )
                                        print(f"   Exception Typ: {type(e)}")
                                        import traceback

                                        print(f"   Traceback: {traceback.format_exc()}")
                                        full_response += (
                                            f"\n**📝 Notiz:** {note_title}\n"
                                        )
                                        full_response += f"(Link-Fehler)\n"
                                else:
                                    # Alte Notiz ohne Storage-Datei
                                    print(
                                        f"⚠️ Alte Notiz ohne Storage-Link: {note_title}"
                                    )
                                    full_response += f"\n**📝 Notiz:** {note_title}\n"
                                    full_response += f"(Nur in Datenbank gespeichert)\n"

                            except Exception as e:
                                print(
                                    f"⚠️ Fehler beim Verarbeiten der Notiz-Metadaten: {e}"
                                )
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
                                    res = client.storage.from_(
                                        "privatedocs"
                                    ).create_signed_url(
                                        fn,
                                        3600,
                                        {
                                            "download": True,
                                            "transform": {
                                                "format": "text",
                                                "quality": 100,
                                            },
                                        },
                                    )
                                except Exception:
                                    # Fallback: Standard signed URL
                                    res = client.storage.from_(
                                        "privatedocs"
                                    ).create_signed_url(fn, 3600)

                                signed_url = res.get("signedURL")
                            except Exception as e:
                                print(
                                    f"⚠️ Fehler beim Erstellen der signed URL für TXT-Datei {fn}: {e}"
                                )
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
                                res = client.storage.from_(
                                    "privatedocs"
                                ).create_signed_url(fn, 3600)
                                signed_url = res.get("signedURL")
                                if sorted_pages and signed_url:
                                    signed_url += f"#page={sorted_pages[0]}"
                            except Exception as e:
                                signed_url = "#"

                            full_response += (
                                f"\n**📄 Quelle:** {fn}, Seiten {page_list}\n"
                            )
                            full_response += f"[PDF öffnen]({signed_url})\n"

                    # Finale Bereinigung: Entferne nur unerwünschte Agent-generierte 'Quelle:' Zeilen
                    lines = full_response.split("\n")
                    clean_lines = []
                    for line in lines:
                        line_lower = line.lower().strip()
                        # Behalte unsere formatierten Quellen (📄, 📝), entferne nur unformatierte 'Quelle:' Zeilen
                        if line_lower.startswith("quelle:") and not (
                            "📄" in line or "📝" in line
                        ):
                            continue
                        clean_lines.append(line)

                    final_response = "\n".join(clean_lines).strip()
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

                                        # 💾 Chat-Historie speichern (Service-Client für alle Chat-berechtigten User)
                    try:
                        sb = get_db_client_for_chat_operations()
                        # Versuche zuerst mit user_id, bei Fehler ohne user_id
                        chat_record = {
                            "user_name": st.session_state["user"].email or "user",
                            "question": user_input,
                            "answer": final_response,
                        }
                        
                        try:
                            # Versuche mit user_id für neuere Schema-Versionen
                            chat_record["user_id"] = st.session_state["user"].id
                            sb.table("chat_history").insert(chat_record).execute()
                        except Exception as schema_error:
                            if "user_id" in str(schema_error):
                                print("⚠️ user_id Spalte nicht verfügbar, verwende Legacy-Schema")
                                # Entferne user_id und versuche nochmal
                                chat_record.pop("user_id", None)
                                sb.table("chat_history").insert(chat_record).execute()
                            else:
                                raise schema_error
                    except Exception as e:
                        print(f"⚠️ Fehler beim Speichern der Chat-Historie: {e}")


                st.markdown("</div>", unsafe_allow_html=True)
            # Kein else-Block nötig - ohne Input wird einfach nichts angezeigt

        with chat_tab2:
            # Chat Historie implementiert
            render_chat_history()

    # Add-Tab (zweiter Tab wenn verfügbar)
    if has_role("data_user", "admin"):
        tab_index = 0
        if has_role("chatbot_user", "data_user", "admin"):
            tab_index = 1
        with tabs[tab_index]:
            # Erstelle Untermenü mit zwei Optionen (ohne doppelten Titel)
            knowledge_tab1, knowledge_tab2 = st.tabs(
                ["Notiz hinzufügen", "Dokumente hochladen"]
            )

        with knowledge_tab1:
            st.markdown(
                "Du kannst hier eigene Notizen, Feedback oder Empfehlungen eintragen, die sofort durchsuchbar sind."
            )

            # Initialisierung für Erfolgs-/Fehlermeldungen und Reset-Counter
            if "note_save_message" not in st.session_state:
                st.session_state.note_save_message = ""
            if "input_reset_counter" not in st.session_state:
                st.session_state.input_reset_counter = 0

            # Eingabefelder mit dynamischen Keys für Reset-Funktionalität
            reset_suffix = st.session_state.input_reset_counter
            manual_title = st.text_input(
                "🏷️ Überschrift",
                key=f"manual_title_input_{reset_suffix}",
            )
            manual_text = st.text_area(
                "✍️ Dein Wissen", 
                key=f"manual_text_input_{reset_suffix}"
            )

            # Handle manuelle Quelle sicher
            source_options = ["Wissen", "Beratung", "Meeting", "Feedback", "Sonstiges"]
            source_type = st.selectbox(
                "Kategorie",
                source_options,
                index=1,  # Default zu "Beratung"
                key=f"manual_source_input_{reset_suffix}",
            )

            # Speichern-Button
            if st.button("✅ Wissen / Notiz speichern", key="save_button"):
                    # Clear any previous messages and table
                    st.session_state.note_save_message = ""
                    if "note_processing_table" in st.session_state:
                        del st.session_state.note_processing_table

                    if not manual_title.strip() or not manual_text.strip():
                        st.warning(
                            "⚠️ Bitte gib sowohl eine Überschrift als auch einen Text ein."
                        )
                    else:
                        # Initialize processing table
                        table_data = [{
                            "Titel": manual_title.strip(),
                            "Fortschritt": "0%", 
                            "Status": "⏳ Wartend"
                        }]
                        
                        # Create table placeholder
                        table_placeholder = st.empty()
                        table_placeholder.table(table_data)
                        
                        def update_note_table(progress, status):
                            table_data[0]["Fortschritt"] = progress
                            table_data[0]["Status"] = status
                            table_placeholder.table(table_data)
                            st.session_state.note_processing_table = table_data.copy()

                        try:
                            # Step 1: Initial validation
                            update_note_table("5%", "🔄 Prüfung...")
                            
                            sb = get_db_client_for_upload_operations()
                            existing = (
                                sb.table("rag_pages")
                                .select("url")
                                .ilike("url", f"{manual_title.strip()}%")
                                .execute()
                            )
                            if existing.data:
                                update_note_table("Bereits vorhanden", "⚠️ Übersprungen")
                                st.warning(
                                    f"⚠️ Ein Eintrag mit der Überschrift '{manual_title.strip()}' existiert bereits."
                                )
                            else:
                                # Step 2: Prepare filename
                                update_note_table("10%", "🔄 Vorbereitung...")
                                
                                # Use appropriate client (Service for data_user and admin, User for others)
                                sb_user = get_db_client_for_upload_operations()
                                pipeline = DocumentIngestionPipeline(db_client=sb_user)
                                tz_berlin = pytz.timezone("Europe/Berlin")
                                now_berlin = datetime.now(tz_berlin)
                                timestamp = now_berlin.strftime("%Y-%m-%d %H:%M")

                                # Erstelle eine Textdatei für die Notiz (bereinige Dateinamen für Supabase)
                                import re
                                import unicodedata

                                # Schritt 1: Normalisiere Unicode und entferne Akzente/Umlaute
                                normalized = unicodedata.normalize(
                                    "NFD", manual_title.strip()
                                )
                                ascii_title = "".join(
                                    c
                                    for c in normalized
                                    if unicodedata.category(c) != "Mn"
                                )

                                # Schritt 2: Erlaube nur Supabase-konforme Zeichen: a-zA-Z0-9_.-
                                safe_title = re.sub(
                                    r"[^a-zA-Z0-9\s\-_.]", "", ascii_title
                                )

                                # Schritt 3: Ersetze Leerzeichen durch Unterstriche
                                safe_title = re.sub(r"\s+", "_", safe_title)

                                # Schritt 4: Entferne mehrfache Unterstriche
                                safe_title = re.sub(r"_+", "_", safe_title)

                                # Schritt 5: Entferne führende/trailing Unterstriche
                                safe_title = safe_title.strip("_")

                                note_filename = f"{safe_title}_{now_berlin.strftime('%Y%m%d_%H%M')}.txt"
                                note_content = f"Titel: {manual_title.strip()}\nQuelle: {source_type}\nErstellt: {timestamp}\n\n{manual_text}"

                                # Step 3: Storage upload
                                update_note_table("20%", "📁 Storage-Upload...")
                                
                                storage_success = False
                                try:
                                    # UTF-8 mit BOM für bessere Browser-Kompatibilität bei deutschen Umlauten
                                    note_content_bytes = "\ufeff".encode(
                                        "utf-8"
                                    ) + note_content.encode("utf-8")

                                    supabase_client.client.storage.from_(
                                        "privatedocs"
                                    ).upload(
                                        note_filename,
                                        note_content_bytes,
                                        {
                                            "cacheControl": "3600",
                                            "x-upsert": "true",
                                            "content-type": "text/plain; charset=utf-8",
                                        },
                                    )
                                    print(
                                        f"✅ Notiz im Storage gespeichert: {note_filename}"
                                    )
                                    storage_success = True
                                except Exception as storage_error:
                                    print(
                                        f"⚠️ Storage-Upload fehlgeschlagen: {storage_error}"
                                    )
                                    update_note_table("20%", "⚠️ Storage-Fehler (wird trotzdem fortgesetzt)")
                                    # Kurz warten damit User die Meldung sieht
                                    time.sleep(1)
                                    
                                # Step 4: Prepare metadata
                                update_note_table("30%", "🔄 Metadaten...")
                                
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
                                        "is_manual": True,  # Flag für manuelle Notizen
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
                                        "is_manual": True,  # Flag für manuelle Notizen
                                    }
                                
                                # Step 5: Text processing (Chunking + Embeddings + DB)
                                update_note_table("40%", "🧠 Chunking & Embeddings...")
                                
                                result = pipeline.process_text(
                                    content=manual_text,
                                    metadata=metadata,
                                    url=manual_title.strip(),
                                )
                                
                                # Step 6: Final result
                                if result and result.get("status") == "duplicate":
                                    update_note_table("Identisches Duplikat", "ℹ️ Übersprungen")
                                    st.session_state.note_save_message = (
                                        "ℹ️ Identische Notiz bereits vorhanden - nicht erneut gespeichert"
                                    )
                                else:
                                    chunks_count = result.get("chunks_count", 0) if result else 0
                                    update_note_table(f"✅ {chunks_count} Chunks erstellt", "✅ Erfolgreich")
                                    st.session_state.note_save_message = (
                                        "✅ Wissen/Notizen erfolgreich gespeichert"
                                    )
                                
                                await update_available_sources()
                                
                                # Lösche Input-Felder nach erfolgreichem Speichern nur bei Erfolg
                                if not (result and result.get("status") == "duplicate"):
                                    # Clear table after successful processing
                                    if "note_processing_table" in st.session_state:
                                        del st.session_state.note_processing_table
                                    st.session_state.input_reset_counter += 1
                                    st.rerun()
                                
                        except Exception as e:
                            # Set error message and update table
                            error_msg = str(e)
                            if "duplicate" in error_msg.lower():
                                update_note_table("Duplikat gefunden", "⚠️ Übersprungen")
                                st.session_state.note_save_message = "ℹ️ Identische Notiz bereits vorhanden"
                            else:
                                update_note_table(f"Fehler: {error_msg[:30]}...", "❌ Fehlgeschlagen") 
                                st.session_state.note_save_message = f"❌ Fehler beim Speichern: {error_msg}"


            # Always display persistent note processing table if it exists
            if "note_processing_table" in st.session_state and st.session_state.note_processing_table:
                st.subheader("📊 Notiz-Verarbeitung")
                st.table(st.session_state.note_processing_table)

            # Display success/error message below buttons
            if st.session_state.note_save_message:
                if st.session_state.note_save_message.startswith("✅"):
                    st.success(st.session_state.note_save_message)
                elif st.session_state.note_save_message.startswith("❌"):
                    st.error(st.session_state.note_save_message)
                else:
                    st.info(st.session_state.note_save_message)

        with knowledge_tab2:
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
                    (f, f"{f.name}_{hash(f.getvalue().hex())}") for f in uploaded_files
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
                    keys_to_remove = [
                        key
                        for key in st.session_state.keys()
                        if key.startswith("selection_")
                    ]
                    for key in keys_to_remove:
                        del st.session_state[key]
                    st.session_state.last_file_selection = current_file_names

                # Separate new files from already processed ones
                new_files = [
                    (f, file_id)
                    for f, file_id in all_uploaded_files
                    if file_id not in st.session_state.processed_files
                ]

                already_processed_files = [
                    (f, file_id)
                    for f, file_id in all_uploaded_files
                    if file_id in st.session_state.processed_files
                ]

                # Only process if we have new files to upload
                if new_files and not st.session_state.get("currently_uploading", False):
                    # Set upload in progress flag
                    st.session_state.currently_uploading = True

                    # Clear old upload status table for new upload
                    st.session_state.upload_status_table = []

                    # Create initial table data for ALL files (new and already processed)
                    table_data = []

                    # Add new files to table
                    for uploaded_file, file_id in new_files:
                        # Check file type first
                        file_ext = (
                            uploaded_file.name.lower().split(".")[-1]
                            if "." in uploaded_file.name
                            else ""
                        )
                        if file_ext not in ["pdf", "txt"]:
                            table_data.append(
                                {
                                    "Dateiname": uploaded_file.name,
                                    "Fortschritt": "Ungültiger Dateityp",
                                    "Status": "❌ Error",
                                }
                            )
                        else:
                            table_data.append(
                                {
                                    "Dateiname": uploaded_file.name,
                                    "Fortschritt": "0%",
                                    "Status": "⏳ Wartend",
                                }
                            )

                    # Add already processed files to table
                    for uploaded_file, file_id in already_processed_files:
                        table_data.append(
                            {
                                "Dateiname": uploaded_file.name,
                                "Fortschritt": "Bereits in dieser Session verarbeitet",
                                "Status": "✅ Bereits hochgeladen",
                            }
                        )

                    # Create table placeholder
                    table_placeholder = st.empty()

                    # Display initial table
                    table_placeholder.table(table_data)

                    # Process each NEW file (skip already processed ones)
                    for i, (uploaded_file, file_id) in enumerate(new_files):
                        # Helper function to update table
                        def update_table_row(filename, progress, status):
                            for row in table_data:
                                if row["Dateiname"] == filename:
                                    row["Fortschritt"] = progress
                                    row["Status"] = status
                                    break
                            table_placeholder.table(table_data)

                        # Check file type first
                        file_ext = (
                            uploaded_file.name.lower().split(".")[-1]
                            if "." in uploaded_file.name
                            else ""
                        )
                        if file_ext not in ["pdf", "txt"]:
                            update_table_row(
                                uploaded_file.name,
                                f"Nur PDF und TXT erlaubt",
                                "❌ Error",
                            )
                            continue

                        safe_filename = sanitize_filename(uploaded_file.name)
                        update_table_row(uploaded_file.name, "5%", "🔄 Prüfung...")

                        try:
                            # Check file size (200MB limit as per UI)
                            if uploaded_file.size > 200 * 1024 * 1024:  # 200MB
                                update_table_row(
                                    uploaded_file.name,
                                    "Datei zu groß (>200MB)",
                                    "❌ Error",
                                )
                                continue

                            file_bytes = uploaded_file.getvalue()

                            # Check if file is empty
                            if len(file_bytes) == 0:
                                update_table_row(
                                    uploaded_file.name, "Datei ist leer", "❌ Error"
                                )
                                continue

                            file_hash = compute_file_hash(file_bytes)

                            # 🔍 Duplikatprüfung anhand Hash
                            sb = get_db_client_for_upload_operations()
                            existing_hash = (
                                sb.table("rag_pages")
                                .select("id")
                                .eq("metadata->>file_hash", file_hash)
                                .execute()
                            )

                            if existing_hash.data:
                                update_table_row(
                                    uploaded_file.name,
                                    "Bereits vorhanden (Hash-Duplikat)",
                                    "⚠️ Übersprungen",
                                )
                                continue

                            # ✅ Duplikatprüfung vor Upload
                            sb = get_db_client_for_upload_operations()
                            existing = (
                                sb.table("rag_pages")
                                .select("id")
                                .eq("url", safe_filename)
                                .execute()
                            )

                            if existing.data:
                                update_table_row(
                                    uploaded_file.name,
                                    "Bereits in Datenbank vorhanden",
                                    "⚠️ Übersprungen",
                                )
                                continue

                        except Exception as e:
                            update_table_row(
                                uploaded_file.name,
                                f"Fehler bei Prüfung: {str(e)}",
                                "❌ Error",
                            )
                            continue

                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=Path(uploaded_file.name).suffix
                        ) as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_file_path = temp_file.name

                        try:
                            update_table_row(
                                uploaded_file.name, "10%", "📥 Upload startet..."
                            )

                            # Content-Type dynamisch bestimmen
                            mime_type, _ = mimetypes.guess_type(safe_filename)
                            if not mime_type:
                                mime_type = "application/octet-stream"

                            # Für TXT-Dateien explizit UTF-8 Encoding setzen
                            if safe_filename.lower().endswith(".txt"):
                                mime_type = "text/plain; charset=utf-8"

                            # Storage upload with error handling
                            storage_success = False
                            storage_error_msg = ""

                            # Für TXT-Dateien spezielles UTF-8 Handling
                            if safe_filename.lower().endswith(".txt"):
                                # TXT-Dateien als UTF-8 Text lesen und als UTF-8 Bytes mit BOM hochladen
                                try:
                                    with open(
                                        temp_file_path, "r", encoding="utf-8"
                                    ) as f:
                                        content = f.read()

                                    # UTF-8 mit BOM für bessere Browser-Kompatibilität bei deutschen Umlauten (wie bei Notizen)
                                    content_bytes = "\ufeff".encode(
                                        "utf-8"
                                    ) + content.encode("utf-8")

                                    supabase_client.client.storage.from_(
                                        "privatedocs"
                                    ).upload(
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
                                    # Fallback: Als Bytes hochladen
                                    try:
                                        with open(temp_file_path, "rb") as f:
                                            supabase_client.client.storage.from_(
                                                "privatedocs"
                                            ).upload(
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
                                        storage_error_msg = (
                                            f"Fallback-Fehler: {str(final_error)}"
                                        )
                            else:
                                # Andere Dateitypen normal als Bytes hochladen
                                try:
                                    with open(temp_file_path, "rb") as f:
                                        supabase_client.client.storage.from_(
                                            "privatedocs"
                                        ).upload(
                                            safe_filename,
                                            f.read(),
                                            {
                                                "cacheControl": "3600",
                                                "x-upsert": "true",
                                                "content-type": mime_type,
                                            },
                                        )
                                        storage_success = True
                                except Exception as storage_error:
                                    storage_error_msg = str(storage_error)

                            if not storage_success:
                                update_table_row(
                                    uploaded_file.name,
                                    f"Storage-Fehler: {storage_error_msg}",
                                    "❌ Error",
                                )
                                continue

                            update_table_row(
                                uploaded_file.name, "50%", "🧠 Verarbeitung..."
                            )

                            # ⇩ HIER: Erstelle Signed-URL und erweiterte Metadaten
                            try:
                                signed_url_response = supabase_client.client.storage.from_("privatedocs").create_signed_url(safe_filename, 3600)
                                signed_url = signed_url_response.get("signedURL", "")
                                print(f"✅ Signed URL erstellt: {signed_url[:50]}...")
                            except Exception as e:
                                print(f"⚠️ Fehler beim Erstellen der Signed URL: {e}")
                                signed_url = ""

                            # Bestimme doc_type basierend auf Dateiendung
                            file_extension = safe_filename.lower().split('.')[-1] if '.' in safe_filename else ''
                            if file_extension == 'pdf':
                                doc_type = "application/pdf"
                            elif file_extension == 'txt':
                                doc_type = "text/plain"
                            else:
                                doc_type = "application/octet-stream"

                            # Erweiterte Metadaten für bessere Dokumentenverwaltung
                            metadata = {
                                "title": os.path.splitext(safe_filename)[0],  # Dateiname ohne Endung als Titel
                                "source_url": safe_filename,  # Storage-Pfad als source_url
                                "doc_type": doc_type,  # MIME-Type basierend auf Dateiendung
                                "original_filename": safe_filename,
                                "signed_url": signed_url,  # Signed URL für direkten Zugriff
                                "file_hash": file_hash,
                                "source": "ui_upload",
                                "upload_time": str(datetime.now()),
                                "source_filter": "privatedocs",
                            }

                            def on_phase(phase: str, processed: int, total: int):
                                """Update progress table based on processing phase."""
                                try:
                                    if phase == "chunk":
                                        # P2: Chunking phase (agentisch oder klassisch)
                                        if total > 0:
                                            progress = max(60, min(75, 60 + (processed / total) * 15))
                                            update_table_row(uploaded_file.name, f"{progress:.0f}%", f"🔪 Chunking ({processed}/{total})...")
                                        else:
                                            update_table_row(uploaded_file.name, "60%", "🔪 Chunking...")
                                    
                                    elif phase == "embed":
                                        # P3: Embedding generation
                                        if total > 0:
                                            progress = max(75, min(90, 75 + (processed / total) * 15))
                                            update_table_row(uploaded_file.name, f"{progress:.0f}%", f"🧠 Embeddings ({processed}/{total})...")
                                        else:
                                            update_table_row(uploaded_file.name, "75%", "🧠 Embeddings...")
                                    
                                    elif phase == "db":
                                        # P4: Database insertion
                                        if total > 0:
                                            progress = max(90, min(98, 90 + (processed / total) * 8))
                                            update_table_row(uploaded_file.name, f"{progress:.0f}%", f"💾 DB-Speicherung ({processed}/{total})...")
                                        else:
                                            update_table_row(uploaded_file.name, "90%", "💾 DB-Speicherung...")
                                    
                                    elif phase == "finalize":
                                        # P5: Finalization (from process_document)
                                        if processed == 0:
                                            update_table_row(uploaded_file.name, "98%", "🔧 Finalisierung...")
                                        else:
                                            update_table_row(uploaded_file.name, "99%", "✅ Abschluss...")
                                    
                                    else:
                                        # Unknown phase - log for debugging
                                        print(f"🐛 Unknown phase: {phase} ({processed}/{total})")
                                        progress = max(50, min(98, 50 + (processed / max(total, 1)) * 45))
                                        update_table_row(uploaded_file.name, f"{progress:.0f}%", f"🔄 {phase.title()}...")
                                        
                                except Exception as e:
                                    print(f"⚠️ Progress update error for {uploaded_file.name}: {e}")
                                    # Continue processing even if UI update fails

                            result = await process_document(
                                temp_file_path,
                                safe_filename,
                                metadata,  # ⇦ Jetzt mit erweiterten Metadaten
                                on_phase=on_phase,
                            )

                            # Final status update
                            if result["success"]:
                                update_table_row(
                                    uploaded_file.name,
                                    f'✅ {result["chunk_count"]} Textabschnitte',
                                    "✅ Hochgeladen",
                                )
                                st.session_state.processed_files.add(file_id)
                            else:
                                update_table_row(
                                    uploaded_file.name,
                                    f'Fehler: {result.get("error","Unbekannt")}',
                                    "❌ Error",
                                )

                        except Exception as e:
                            update_table_row(
                                uploaded_file.name,
                                f"Unerwarteter Fehler: {str(e)}",
                                "❌ Error",
                            )
                            print(
                                f"❌ Unerwarteter Fehler beim Verarbeiten von {uploaded_file.name}: {e}"
                            )
                        finally:
                            if "temp_file_path" in locals():
                                try:
                                    os.unlink(temp_file_path)
                                except:
                                    pass  # Ignore cleanup errors

                    # Store table in session state for persistence
                    st.session_state.upload_status_table = table_data
                    st.session_state.just_uploaded = True

                    # Clear the live update table to avoid duplicates
                    table_placeholder.empty()

                    # Reset upload flags
                    st.session_state.just_uploaded = True
                    st.session_state.currently_uploading = False

                    # Aktualisiere Quellen explizit nach Upload
                    await update_available_sources()
                    print(
                        f"🔄 Nach Upload: {st.session_state.get('document_count', 0)} Dokumente, {st.session_state.get('knowledge_count', 0)} Notizen"
                    )

                    # Count successful uploads for rerun logic only
                    successful_uploads = sum(
                        1 for row in table_data if row["Status"] == "✅ Hochgeladen"
                    )

                    # Update header by triggering rerun only if files were successfully uploaded
                    if successful_uploads > 0:
                        # Set flag to prevent "already processed" message after rerun
                        st.session_state.upload_just_completed = True
                        st.rerun()

                elif (
                    already_processed_files
                    and not new_files
                    and not st.session_state.get("upload_just_completed", False)
                ):
                    st.info("Alle Dateien wurden bereits verarbeitet")

            # Always display persistent upload status table if it exists
            if (
                "upload_status_table" in st.session_state
                and st.session_state.upload_status_table
                and not st.session_state.get("currently_uploading", False)
            ):
                st.subheader("📊 Upload-Status")
                st.table(st.session_state.upload_status_table)

            st.markdown(
                "<hr style='margin-top: 6px; margin-bottom: 6px;'>",
                unsafe_allow_html=True,
            )


    # Del-Tab (vorletzter Tab wenn verfügbar) 
    if has_role("data_user", "admin"):
        # Berechne Index: letzter Tab wenn nur delete, vorletzter wenn auch admin-Tab da ist
        if has_role("admin"):
            tab_index = len(tabs) - 2  # Vorletzter Tab (vor admin)
        else:
            tab_index = len(tabs) - 1  # Letzter Tab
        with tabs[tab_index]:
            await render_delete_preview_ui()

    # Benutzerverwaltungs-Tab (letzter Tab - nur für admin)
    if has_role("admin"):
        tab_index = len(tabs) - 1  # Letzter Tab
        with tabs[tab_index]:
            render_user_management()


if __name__ == "__main__":
    asyncio.run(main())
