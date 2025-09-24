from supabase import create_client
import jwt,os
from dotenv import load_dotenv

load_dotenv()  # liest .env (optional)

url = os.getenv("SUPABASE_URL", "https://dpmsxbzcesncliysjirw.supabase.co")
srk = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # NICHT ins Repo committen!

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

email="roedenbeck@hotmail.com"; password="abc100"
sess = sb.auth.sign_in_with_password({"email": email, "password": password}).session
claims = jwt.decode(sess.access_token, options={"verify_signature": False})
print("Info:",claims.get("app_metadata", {}))  # <- hier sollte roles: [...] drinstehen
