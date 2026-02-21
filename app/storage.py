from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

async def store_voice_embedding(user_id: str, embedding):
    # convert anything -> JSON-friendly list
    try:
        embedding = embedding.tolist()
    except:
        embedding = list(embedding)

    data = (
        supabase.table("biometrics")
        .upsert({
            "id": user_id,
            "biometrics": embedding
        })
        .execute()
    )

    return data.data

