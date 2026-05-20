import sys
import io
from pathlib import Path

# Force UTF-8 output so rupee symbol and other Unicode print correctly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from app.retriever import load_store, retrieve, rerank, generate

ROOT = Path(__file__).parent.parent
store = load_store(str(ROOT / "data" / "vector_store.json"))

questions = [
    "What is the fee for the foundation level?",
    "How many credits are needed for the BS degree?",
    "What happens if I fail a course?",
    "Can I transfer credits from NPTEL courses?",
    "What are the eligibility criteria for the qualifier exam?",
]

for i, q in enumerate(questions, 1):
    print(f"\n{'='*70}")
    print(f"Q{i}: {q}")
    print("="*70)
    candidates = retrieve(q, store, top_k=5)
    reranked   = rerank(q, candidates, top_n=3)
    answer     = generate(q, reranked)
    print(f"Sources: {[c['heading'] for c in reranked]}")
    print(f"\nAnswer:\n{answer}")
