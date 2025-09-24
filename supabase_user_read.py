from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()  # liest .env (optional)

url = os.getenv("SUPABASE_URL", "https://dpmsxbzcesncliysjirw.supabase.co")
srk = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # NICHT ins Repo committen!

uid="b4dc43db-e7a2-4c10-b111-63b3c579240b"  # deine UID
c=create_client(url,srk)
u=c.auth.admin.get_user_by_id(uid)
print("Info:",u.user.app_metadata)   # <- hier sollten die roles auftauchen

