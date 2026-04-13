"""MedRAG FastAPI."""
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator
import psycopg

from database import init_db, insert_user, get_user, log_query, fetch_history
from auth import hash_pw, verify_pw, create_token, current_user
from rag_chain import get_pipeline

app = FastAPI(title="MedRAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Instrumentator().instrument(app).expose(app)

class RegisterIn(BaseModel):
    username: str
    password: str

class QueryIn(BaseModel):
    question: str

@app.on_event("startup")
def _startup():
    init_db()
    get_pipeline()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/register")
def register(body: RegisterIn):
    if get_user(body.username):
        raise HTTPException(400, "Username already exists")
    try:
        uid = insert_user(body.username, hash_pw(body.password))
    except psycopg.Error as e:
        raise HTTPException(500, f"DB error: {e}")
    return {"id": uid, "username": body.username}

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    u = get_user(form.username)
    if not u or not verify_pw(form.password, u["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": create_token({"sub": u["username"], "uid": u["id"]}), "token_type": "bearer"}

@app.post("/query")
def query(body: QueryIn, user=Depends(current_user)):
    if not body.question.strip():
        raise HTTPException(400, "Empty question")
    result = get_pipeline().answer(body.question)
    log_query(user_id=user["id"], username=user["username"], question=body.question,
              answer=result["answer"], sources=result["sources"], score=result["classifier_score"],
              label=result["classifier_label"], latency_ms=result["latency_ms"], model=result["model"])
    return result

@app.get("/history")
def history(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0), user=Depends(current_user)):
    return {"items": fetch_history(user["id"], limit=limit, offset=offset), "limit": limit, "offset": offset}
