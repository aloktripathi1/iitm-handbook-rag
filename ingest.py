import os
import json
import time
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

DOC_URL = "https://docs.google.com/document/d/e/2PACX-1vRxGnnDCVAO3KX2CGtMIcJQuDrAasVk2JHbDxkjsGrTP5ShhZK8N6ZSPX89lexKx86QPAUswSzGLsOA/pub"
OUTPUT_PATH = "vector_store.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_WORDS = 600
MIN_WORDS = 50


def fetch_html(url: str) -> str:
    print(f"Fetching document from {url} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    print(f"  Fetched {len(response.content)} bytes")
    return response.text


def extract_chunks(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    chunks = []
    heading_tags = {"h1", "h2", "h3"}

    current_heading = "Introduction"
    current_paragraphs = []

    def flush(heading: str, paragraphs: list[str]) -> None:
        if not paragraphs:
            return
        text = "\n".join(paragraphs).strip()
        words = text.split()
        if len(words) < MIN_WORDS:
            return

        if len(words) <= MAX_WORDS:
            chunks.append({"heading": heading, "text": text})
            print(f"  Chunk [{len(chunks)}]: '{heading[:60]}' — {len(words)} words")
        else:
            sub_chunks: list[list[str]] = []
            current_sub: list[str] = []
            current_count = 0

            for para in paragraphs:
                para_words = len(para.split())
                if current_count + para_words > MAX_WORDS and current_sub:
                    sub_chunks.append(current_sub)
                    current_sub = [para]
                    current_count = para_words
                else:
                    current_sub.append(para)
                    current_count += para_words

            if current_sub:
                sub_chunks.append(current_sub)

            for i, sub in enumerate(sub_chunks):
                sub_text = "\n".join(sub).strip()
                sub_words = sub_text.split()
                if len(sub_words) < MIN_WORDS:
                    continue
                sub_heading = f"{heading} (part {i + 1})" if len(sub_chunks) > 1 else heading
                chunks.append({"heading": sub_heading, "text": sub_text})
                print(f"  Chunk [{len(chunks)}]: '{sub_heading[:60]}' — {len(sub_words)} words")

    body = soup.find("body") or soup
    for element in body.find_all(True):
        tag = element.name
        if tag in heading_tags:
            flush(current_heading, current_paragraphs)
            current_heading = element.get_text(separator=" ", strip=True)
            current_paragraphs = []
        elif tag == "p":
            text = element.get_text(separator=" ", strip=True)
            if text:
                current_paragraphs.append(text)
        elif tag in {"li", "td", "th"}:
            text = element.get_text(separator=" ", strip=True)
            if text:
                current_paragraphs.append(text)

    flush(current_heading, current_paragraphs)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    print(f"\nLoading embedding model '{EMBED_MODEL}' ...")
    try:
        model = SentenceTransformer(EMBED_MODEL)
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {e}") from e

    texts = [f"{c['heading']}\n\n{c['text']}" for c in chunks]
    print(f"Embedding {len(texts)} chunks in one batch call ...")

    try:
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    except Exception as e:
        raise RuntimeError(f"Embedding call failed: {e}") from e

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    print(f"  Embeddings done — dim={embeddings.shape[1]}")
    return chunks


def save_store(chunks: list[dict], path: str) -> None:
    records = [
        {
            "id": i,
            "heading": c["heading"],
            "text": c["text"],
            "embedding": c["embedding"],
        }
        for i, c in enumerate(chunks)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"\nSaved {len(records)} chunks to {path}")


def estimate_tokens(chunks: list[dict]) -> int:
    total_chars = sum(len(c["text"]) + len(c["heading"]) for c in chunks)
    return total_chars // 4


def main() -> None:
    start = time.time()

    html = fetch_html(DOC_URL)

    print("\nParsing and chunking document ...")
    chunks = extract_chunks(html)
    print(f"\nTotal chunks created: {len(chunks)}")

    estimated_tokens = estimate_tokens(chunks)
    print(f"Estimated tokens (text only, ~4 chars/token): {estimated_tokens:,}")

    chunks = embed_chunks(chunks)
    save_store(chunks, OUTPUT_PATH)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Chunks:           {len(chunks)}")
    print(f"  Tokens estimated: {estimated_tokens:,}")
    print(f"  Output:           {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
