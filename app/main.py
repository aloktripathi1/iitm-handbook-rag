from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.retriever import generate, load_store, rerank, retrieve

store: list[dict] = []

ROOT_DIR = Path(__file__).parent.parent
STATIC_DIR = ROOT_DIR / "static"
DATA_DIR = ROOT_DIR / "data"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    print("Loading vector store ...")
    try:
        store = load_store(str(DATA_DIR / "vector_store.json"))
        print(f"  Loaded {len(store)} chunks")
    except FileNotFoundError:
        print("  WARNING: vector_store.json not found — run scripts/ingest.py first")
    yield


app = FastAPI(title="HandbookGPT", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_retrieved: int
    chunks_after_rerank: int


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not store:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not loaded. Run scripts/ingest.py first.",
        )
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        candidates = retrieve(request.question, store, top_k=10)
        reranked = rerank(request.question, candidates, top_n=3)
        answer = generate(request.question, reranked)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    sources = list(dict.fromkeys(c["heading"] for c in reranked))

    return ChatResponse(
        answer=answer,
        sources=sources,
        chunks_retrieved=len(candidates),
        chunks_after_rerank=len(reranked),
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "chunks_loaded": len(store)}
