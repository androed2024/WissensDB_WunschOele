from supabase import create_client
import os

from dotenv import load_dotenv

load_dotenv()  # liest .env (optional)

url = os.getenv("SUPABASE_URL", "https://dpmsxbzcesncliysjirw.supabase.co")
srk = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # NICHT ins Repo committen!

c = create_client(url,srk)

USER_ID = "b4dc43db-e7a2-4c10-b111-63b3c579240b"          # deine UID
c.auth.admin.update_user_by_id(USER_ID, {"password": "Abc100!Test"})
print("Temp-Passwort gesetzt")
