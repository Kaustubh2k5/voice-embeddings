import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    supabase_url: str = os.getenv("SUPABASE_URL")
    supabase_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    jwt_secret: str = os.getenv("SUPABASE_JWT_SECRET")

    def validate(self):
        if not self.supabase_url or not self.supabase_key or not self.jwt_secret:
            raise RuntimeError("Missing Supabase environment variables")

settings = Settings()
settings.validate()
