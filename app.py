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
            "🗑️ Dokument / Notiz löschen",
        ]
    )

    with tab1:
        st.markdown(
            "<h4>💬 Spreche mit dem Wunsch-Öle KI Assistenten</h4>",
            unsafe_allow_html=True,
        )

        for msg in st.session_state.messages:
            if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
                for part in msg.parts:
                    display_message_part(part)

        user_input = st.chat_input("Stelle eine Frage zu den Dokumenten...")
        if user_input:
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
                note_sources = defaultdict(set)
                
                if hasattr(rag_agent, "last_match") and rag_agent.last_match:
                    print("--- Treffer im Retrieval ---")
                    # Score-basierte Filterung für Anzeige
                    DISPLAY_MIN_SIM = float(os.getenv("RAG_DISPLAY_MIN_SIM", "0.50"))
                    
                    for match in rag_agent.last_match:
                        sim = match.get("similarity", 0)
                        # Verwende konfigurierbaren Display-Threshold
                        if sim < DISPLAY_MIN_SIM:
                            print(f"⚠️ Treffer ignoriert (Score {sim:.3f} < {DISPLAY_MIN_SIM})")
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
                                pdf_sources[fn].add(pg)
                                print("✅ Dokument:", fn, "| Seite:", pg, "| Score:", sim)

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
                
                # Füge saubere Quellenangaben hinzu
                if pdf_sources or note_sources:
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
                                        
                                        # Prüfe, ob es eine alte Notiz ist (ohne .txt Extension)
                                        if not original_filename.endswith('.txt') and meta.get("source_filter") == "notes":
                                            # Alte Notiz - keine Storage-Datei verfügbar
                                            note_filename = None
                                            bucket = "notes"
                                        else:
                                            # Neue Notiz - verwende Storage-Dateinamen
                                            note_filename = storage_filename or original_filename
                                            bucket = meta.get("source_filter", "privatedocs")
                                        break
                            
                            # Erstelle signed URL für die Notiz
                            if note_filename and bucket != "notes":
                                # Neue Notiz mit Storage-Datei
                                try:
                                    client = get_supabase_client()
                                    res = client.storage.from_(bucket).create_signed_url(note_filename, 3600)
                                    signed_url = res.get("signedURL", "#")
                                    
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

                # ✅ FINALE Bereinigung: Nur unsere Quellen behalten (📄 oder 📝)
                # Suche nach unseren Markern
                pdf_parts = full_response.split('**📄 Quelle:**')
                note_parts = full_response.split('**📝 Notiz:**')
                
                if len(pdf_parts) > 1:
                    # PDF-Quelle gefunden
                    before_source = pdf_parts[0]
                    our_source_part = pdf_parts[1]
                    
                    # Schneide bei normalem 'Quelle:' ab
                    quelle_pos = our_source_part.lower().find('quelle:')
                    if quelle_pos != -1:
                        our_source_part = our_source_part[:quelle_pos]
                    
                    final_response = (before_source + '**📄 Quelle:**' + our_source_part).strip()
                    
                elif len(note_parts) > 1:
                    # Notiz-Quelle gefunden
                    before_source = note_parts[0]
                    our_source_part = note_parts[1]
                    
                    # Schneide bei normalem 'Quelle:' ab
                    quelle_pos = our_source_part.lower().find('quelle:')
                    if quelle_pos != -1:
                        our_source_part = our_source_part[:quelle_pos]
                    
                    final_response = (before_source + '**📝 Notiz:**' + our_source_part).strip()
                    
                else:
                    # Fallback: Entferne alle 'Quelle:' Zeilen
                    lines = full_response.split('\n')
                    clean_lines = [line for line in lines if not line.lower().strip().startswith('quelle:')]
                    final_response = '\n'.join(clean_lines).strip()
                message_placeholder.markdown(final_response)

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
        source_options = ["Beratung", "Meeting", "Feedback", "Sonstiges"]
        try:
            source_index = source_options.index(st.session_state.manual_source)
        except ValueError:
            source_index = 0

        source_type = st.selectbox(
            "Quelle des Wissens",
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
                            full_title = f"{manual_title.strip()} ({timestamp})"
                            
                            # Erstelle eine Textdatei für die Notiz
                            note_filename = f"{manual_title.strip()}_{now_berlin.strftime('%Y%m%d_%H%M')}.txt"
                            note_content = f"Titel: {manual_title.strip()}\nQuelle: {source_type}\nErstellt: {timestamp}\n\n{manual_text}"
                            
                            # Speichere Notiz im Storage
                            try:
                                supabase_client.client.storage.from_("privatedocs").upload(
                                    note_filename,
                                    note_content.encode('utf-8'),
                                    {
                                        "cacheControl": "3600",
                                        "x-upsert": "true",
                                        "content-type": "text/plain; charset=utf-8",
                                    },
                                )
                                print(f"✅ Notiz im Storage gespeichert: {note_filename}")
                            except Exception as storage_error:
                                print(f"⚠️ Storage-Upload fehlgeschlagen: {storage_error}")
                                # Fahre trotzdem fort - Notiz wird zumindest in der DB gespeichert
                            
                            metadata = {
                                "source": "manuell",
                                "quelle": source_type,
                                "title": manual_title.strip(),
                                "upload_time": now_berlin.isoformat(),
                                "original_filename": note_filename,  # Verwende den Storage-Dateinamen
                                "source_filter": "privatedocs",  # Gleicher Bucket wie PDFs
                                "storage_filename": note_filename,  # Zusätzlich für Klarheit
                            }
                            result = pipeline.process_text(
                                content=manual_text,
                                metadata=metadata,
                                url=full_title,
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
        <div style="padding:1rem;background:#f6f7fa;border-radius:8px;font-size:16px;">
            📎 Dateien für Wissensdatenbank hochladen<br>
            <small>(max. 200 MB pro Datei • PDF oder TXT)</small>
        </div>
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

        if uploaded_files and not st.session_state.just_uploaded:
            new_files = [
                (f, f"{f.name}_{hash(f.getvalue().hex())}")
                for f in uploaded_files
                if f"{f.name}_{hash(f.getvalue().hex())}"
                not in st.session_state.processed_files
            ]

            if new_files:
                st.subheader("⏳ Upload-Fortschritt")
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (uploaded_file, file_id) in enumerate(new_files):
                    safe_filename = sanitize_filename(uploaded_file.name)

                    file_bytes = uploaded_file.getvalue()
                    file_hash = compute_file_hash(file_bytes)

                    # 🔍 Duplikatprüfung anhand Hash
                    existing_hash = (
                        supabase_client.client.table("rag_pages")
                        .select("id")
                        .eq("metadata->>file_hash", file_hash)
                        .execute()
                    )

                    if existing_hash.data:
                        st.warning(
                            f"⚠️ Die Datei **{safe_filename}** wurde bereits (unter anderem Namen) hochgeladen und wird nicht erneut gespeichert."
                        )
                        continue

                    # ✅ Duplikatprüfung vor Upload
                    existing = (
                        supabase_client.client.table("rag_pages")
                        .select("id")
                        .eq("url", safe_filename)
                        .execute()
                    )

                    if existing.data:
                        st.warning(
                            f"⚠️ Die Datei **{safe_filename}** ist bereits in der Wissensdatenbank vorhanden und wurde nicht erneut hochgeladen."
                        )
                        continue

                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(uploaded_file.name).suffix
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name

                    try:
                        progress_bar.progress(0.05)
                        status_text.markdown(
                            f"🟡 **{safe_filename}**: 📥 *Upload startet...*"
                        )

                        # Content-Type dynamisch bestimmen
                        mime_type, _ = mimetypes.guess_type(safe_filename)
                        if not mime_type:
                            mime_type = "application/octet-stream"
                        
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

                        progress_bar.progress(0.3)
                        status_text.markdown(
                            f"🟠 **{safe_filename}**: 📤 *Dateiübertragung abgeschlossen*"
                        )

                        metadata = {
                            "source": "ui_upload",
                            "upload_time": str(datetime.now()),
                            "original_filename": safe_filename,
                            "file_hash": file_hash,
                        }

                        status_text.markdown(
                            f"🔵 **{safe_filename}**: 🧠 *Verarbeitung läuft...*"
                        )

                        result = await process_document(
                            temp_file_path, safe_filename, metadata
                        )

                        progress_bar.progress(0.8)

                        if result["success"]:
                            st.success(
                                f"✅ {uploaded_file.name} verarbeitet: {result['chunk_count']} Textabschnitte"
                            )
                            st.session_state.document_count += 1
                            st.session_state.processed_files.add(file_id)
                        else:
                            st.error(
                                f"❌ Fehler beim Verarbeiten {uploaded_file.name}: {result['error']}"
                            )

                        progress_bar.progress(1.0)
                        status_text.markdown(
                            f"🟢 **{safe_filename}**: ✅ *Verarbeitung abgeschlossen*"
                        )

                    finally:
                        os.unlink(temp_file_path)

                st.session_state.just_uploaded = True
                await update_available_sources()
                st.rerun()

            else:
                st.info("Alle Dateien wurden bereits verarbeitet")

        st.markdown(
            "<hr style='margin-top: 6px; margin-bottom: 6px;'>", unsafe_allow_html=True
        )

    with tab4:
        st.markdown("<h4>🗑️ Dokument / Notiz löschen</h4>", unsafe_allow_html=True)

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
                        st.markdown(f"**Titel:** {metadata.get('title', 'Unbekannt')}")
                        st.markdown(f"**Quelle:** {metadata.get('quelle', '–')}")
                        st.markdown("**Inhalt:**")
                        st.code(content, language="markdown")
                    else:
                        # Original PDF anzeigen
                        try:
                            signed_url = metadata.get("signed_url")
                            if signed_url:
                                st.markdown("**📄 Original-PDF Vorschau:**")
                                st.components.v1.html(
                                    f"""
                                    <iframe src=\"{signed_url}\" width=\"100%\" height=\"600px\" style=\"border:1px solid #ccc; border-radius: 6px;\"></iframe>
                                    """,
                                    height=620,
                                )
                            else:
                                st.warning("Keine Original-PDF verfügbar.")
                        except Exception as e:
                            st.error(
                                f"Fehler beim Laden der vollständigen Vorschau: {e}"
                            )
                else:
                    st.info("Keine Vorschau verfügbar.")

            except Exception as e:
                st.error(f"Fehler beim Laden der Vorschau: {e}")

            if st.button("Ausgewählte Dokument/Notiz löschen"):
                st.write("Dateiname zur Löschung:", delete_filename)

                storage_deleted = db_deleted = False

                try:
                    print("Lösche:", delete_filename)
                    supabase_client.client.storage.from_("privatedocs").remove(
                        [delete_filename]
                    )
                    storage_deleted = True
                except Exception as e:
                    st.error(f"Löschen aus dem Speicher fehlgeschlagen: {e}")

                try:
                    deleted_count = supabase_client.delete_documents_by_filename(
                        delete_filename
                    )
                    st.code(
                        f"🩨 SQL-Delete für '{delete_filename}' – {deleted_count} Einträge entfernt."
                    )
                    db_deleted = True
                except Exception as e:
                    st.error(f"Datenbank-Löschung fehlgeschlagen: {e}")
                    db_deleted = False

                if storage_deleted and db_deleted:
                    st.success("✅ Vollständig gelöscht.")
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

                await update_available_sources()
                st.rerun()

        else:
            st.info("Keine Dokumente/Notizen zur Löschung verfügbar.")


if __name__ == "__main__":
    asyncio.run(main())
