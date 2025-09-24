# pip install supabase
# start:  python supabase_user_rolle.py
# supabase_user_rolle.py
# pip install supabase python-dotenv

import os, sys
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()  # liest .env (optional)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://dpmsxbzcesncliysjirw.supabase.co")
SERVICE_KEY   = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # NICHT ins Repo committen!

USER_ID = "b4dc43db-e7a2-4c10-b111-63b3c579240b"  # <- deine UID hier rein
ROLES   = ["admin"]  # oder ["data_user"], ["chatbot_user"], ...

client = create_client(SUPABASE_URL, SERVICE_KEY)

client.auth.admin.update_user_by_id(
    USER_ID,
    attributes={"app_metadata": {"roles": ROLES}}
)

print("OK: Rollen gesetzt. Nutzer muss sich neu einloggen, damit das JWT aktualisiert wird.")