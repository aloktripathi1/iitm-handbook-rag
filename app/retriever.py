import json
import os
import re

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
# Any model available on openrouter.ai — free tier options work fine
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")

_client: OpenAI | None = None
_embed_model: SentenceTransformer | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in .env")
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def load_store(path: str = "data/vector_store.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    for record in records:
        record["embedding"] = np.array(record["embedding"], dtype=np.float32)
    return records


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def retrieve(query: str, store: list[dict], top_k: int = 5) -> list[dict]:
    model = _get_embed_model()

    try:
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        query_embedding = np.array(query_embedding, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to embed query: {e}") from e

    scored = [
        {
            "id": record["id"],
            "heading": record["heading"],
            "text": record["text"],
            "source": record.get("source", "unknown"),
            "score": cosine_similarity(query_embedding, record["embedding"]),
        }
        for record in store
    ]

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def rerank(query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
    passages = "\n".join(
        f"{i + 1}. [{c['heading']}] {c['text'][:350]}" for i, c in enumerate(candidates)
    )
    prompt = (
        "Rate each passage's relevance to the question on a scale 0-10.\n"
        'Return only JSON: {"scores": [7, 2, 9, 4, 6]}\n'
        f"Question: {query}\n"
        f"Passages:\n{passages}"
    )

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        raise RuntimeError(f"Rerank LLM call failed: {e}") from e

    raw = response.choices[0].message.content.strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Rerank response did not contain valid JSON: {raw!r}")

    try:
        data = json.loads(match.group())
        scores = data["scores"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Could not parse rerank scores from: {raw!r}") from e

    if len(scores) != len(candidates):
        scores = (scores + [0] * len(candidates))[: len(candidates)]

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]


def generate(query: str, context_chunks: list[dict]) -> str:
    context_parts = [
        f"[{c['heading']}]\n{c['text']}" for c in context_chunks
    ]
    context_text = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a helpful assistant for IITM BS Degree students. "
        "Answer only from the provided handbook and grading document sections. "
        "If the answer is not in the context, say so clearly with: "
        "'I don't know based on the provided documents.' "
        "Always cite which section your answer comes from by referencing the section heading."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        raise RuntimeError(f"Generation LLM call failed: {e}") from e

    return response.choices[0].message.content.strip()
