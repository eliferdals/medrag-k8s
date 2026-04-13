"""JWT auth — MLOps projendeki ile tutarlı pattern."""
import os
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = os.getenv("JWT_SECRET", "change-me")
ALGORITHM = "HS256"
EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "60"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2 = OAuth2PasswordBearer(tokenUrl="/login")

def hash_pw(p): return pwd_ctx.hash(p)
def verify_pw(p, h): return pwd_ctx.verify(p, h)

def create_token(data):
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=EXPIRE_MIN)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def current_user(token: str = Depends(oauth2)):
    err = HTTPException(status_code=401, detail="Invalid token")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if "sub" not in payload: raise err
        return {"id": payload["uid"], "username": payload["sub"]}
    except JWTError:
        raise err
