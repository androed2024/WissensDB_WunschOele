#!/usr/bin/env python3
"""
Debug-Script um zu prüfen was mit der Notiz "BOAT SYNTH 2-T" los ist.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from supabase import create_client

def main():
    print("🔍 Debug: Notiz-Suche Problem")
    print("=" * 50)
    
    # Supabase Client direkt erstellen
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        print("❌ Fehler: SUPABASE_URL oder SUPABASE_SERVICE_ROLE_KEY nicht gesetzt")
        return
        
    client = create_client(supabase_url, supabase_key)
    
    # 1. Prüfe alle manuellen Notizen
    print("\n1. 📝 Alle manuellen Notizen:")
    try:
        result = client.table("rag_pages").select("url, metadata").eq("metadata->>source", "manuell").execute()
        notes = result.data or []
        print(f"   Gefunden: {len(notes)} Notizen")
        for note in notes:
            title = note.get("metadata", {}).get("title", "Unbekannt") 
            url = note.get("url", "")
            print(f"   • Titel: '{title}' | URL: '{url}'")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
    
    # 2. Suche explizit nach "BOAT"
    print("\n2. 🔍 Suche nach 'BOAT' im Content:")
    try:
        result = client.table("rag_pages").select("url, content, metadata").ilike("content", "%BOAT%").execute()
        boat_results = result.data or []
        print(f"   Gefunden: {len(boat_results)} Treffer")
        for res in boat_results:
            title = res.get("metadata", {}).get("title", "Unbekannt")
            content_snippet = res.get("content", "")[:100].replace("\n", " ")
            print(f"   • '{title}': {content_snippet}...")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
    
    # 3. Prüfe alle Einträge der letzten Zeit
    print("\n3. 📅 Neueste Einträge (letzte 10):")
    try:
        result = client.table("rag_pages").select("url, metadata, created_at").order("created_at", desc=True).limit(10).execute()
        recent = result.data or []
        print(f"   Gefunden: {len(recent)} neueste Einträge")
        for entry in recent:
            title = entry.get("metadata", {}).get("title", "Unbekannt")
            source = entry.get("metadata", {}).get("source", "unbekannt")
            url = entry.get("url", "")
            created = entry.get("created_at", "")[:19]  # Datum ohne Zeitzone
            print(f"   • {created} | {source} | '{title}' | URL: {url}")
    except Exception as e:
        print(f"   ❌ Fehler: {e}")
        
    # 4. Prüfe Text-Inhalte mit "SYNTH" oder "2-T" 
    print("\n4. 🔍 Suche nach 'SYNTH' oder '2-T':")
    try:
        result1 = client.table("rag_pages").select("url, content, metadata").ilike("content", "%SYNTH%").execute()
        result2 = client.table("rag_pages").select("url, content, metadata").ilike("content", "%2-T%").execute()
        
        synth_results = result1.data or []
        t2_results = result2.data or []
        
        print(f"   'SYNTH': {len(synth_results)} Treffer")
        print(f"   '2-T': {len(t2_results)} Treffer")
        
        all_results = synth_results + t2_results
        for res in all_results[:3]:  # Nur erste 3 anzeigen
            title = res.get("metadata", {}).get("title", "Unbekannt")
            source = res.get("metadata", {}).get("source", "unbekannt")
            content_snippet = res.get("content", "")[:120].replace("\n", " ")
            print(f"   • {source} | '{title}': {content_snippet}...")
            
    except Exception as e:
        print(f"   ❌ Fehler: {e}")

if __name__ == "__main__":
    main()
