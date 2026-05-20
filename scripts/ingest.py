import json
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

SOURCES = [
    {
        "name": "handbook",
        "url": "https://docs.google.com/document/d/e/2PACX-1vRxGnnDCVAO3KX2CGtMIcJQuDrAasVk2JHbDxkjsGrTP5ShhZK8N6ZSPX89lexKx86QPAUswSzGLsOA/pub",
    },
    {
        "name": "grading",
        "url": "https://docs.google.com/document/d/e/2PACX-1vT5PBOz4OH663W0IJPVGVjG_nfmYZGfFI7W1j-6wTLcex13O_7BZmf6a96Q6liO0W-mLZB5hOGZeNNl/pub?urp=gmail_link",
    },
]

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "vector_store.json"
EMBED_MODEL = "all-MiniLM-L6-v2"
MAX_WORDS = 600
MIN_WORDS = 30  # lower threshold for grading doc which has shorter entries


def fetch_html(url: str) -> str:
    print(f"  Fetching {url[:80]} ...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    print(f"  Fetched {len(response.content):,} bytes")
    return response.text


def extract_chunks(html: str, source: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    chunks = []
    heading_tags = {"h1", "h2", "h3"}

    current_heading = "Introduction"
    current_paragraphs: list[str] = []

    def flush(heading: str, paragraphs: list[str]) -> None:
        if not paragraphs:
            return
        text = "\n".join(paragraphs).strip()
        words = text.split()
        if len(words) < MIN_WORDS:
            return

        if len(words) <= MAX_WORDS:
            chunks.append({"heading": heading, "text": text, "source": source})
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
                if len(sub_text.split()) < MIN_WORDS:
                    continue
                sub_heading = f"{heading} (part {i + 1})" if len(sub_chunks) > 1 else heading
                chunks.append({"heading": sub_heading, "text": sub_text, "source": source})

    body = soup.find("body") or soup
    for element in body.find_all(True):
        tag = element.name
        if tag in heading_tags:
            flush(current_heading, current_paragraphs)
            current_heading = element.get_text(separator=" ", strip=True)
            current_paragraphs = []
        elif tag == "p":
            text = element.get_text(separator=" ", strip=True).replace("\xa0", " ").strip()
            if text:
                current_paragraphs.append(text)
        elif tag in {"li", "td", "th"}:
            text = element.get_text(separator=" ", strip=True).replace("\xa0", " ").strip()
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
    print(f"Embedding {len(texts)} chunks ...")

    try:
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    except Exception as e:
        raise RuntimeError(f"Embedding call failed: {e}") from e

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding.tolist()

    print(f"  Embeddings done — dim={embeddings.shape[1]}")
    return chunks


def save_store(chunks: list[dict], path: Path) -> None:
    records = [
        {
            "id": i,
            "heading": c["heading"],
            "text": c["text"],
            "source": c["source"],
            "embedding": c["embedding"],
        }
        for i, c in enumerate(chunks)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    print(f"\nSaved {len(records)} chunks to {path}")


def main() -> None:
    start = time.time()
    all_chunks: list[dict] = []

    for src in SOURCES:
        print(f"\n--- Source: {src['name']} ---")
        html = fetch_html(src["url"])
        chunks = extract_chunks(html, source=src["name"])
        print(f"  Extracted {len(chunks)} chunks from '{src['name']}'")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across all sources: {len(all_chunks)}")

    all_chunks = embed_chunks(all_chunks)
    save_store(all_chunks, OUTPUT_PATH)

    elapsed = time.time() - start
    by_source = {}
    for c in all_chunks:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    print(f"\nDone in {elapsed:.1f}s")
    for src_name, count in by_source.items():
        print(f"  {src_name}: {count} chunks")


if __name__ == "__main__":
    main()
