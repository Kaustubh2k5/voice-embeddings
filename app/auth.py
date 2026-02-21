from fastapi import Header, HTTPException
from jose import jwt, JWTError
from app.config import settings

async def require_user(authorization: str | None = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")

    token = authorization.replace("Bearer ", "").strip()

    try:
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"]
        )
    except JWTError:
        raise HTTPException(401, "Invalid token")

    sub = payload.get("sub")       # 👈 this is the UUID from auth.users.id

    if not sub:
        raise HTTPException(401, "Token missing subject claim")

    return {"id": sub}
