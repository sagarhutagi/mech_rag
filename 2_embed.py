"""
STEP 2: embed.py
Embeds all chunks into ChromaDB with rich metadata.
Handles all 3 sources: textbook, question_bank, solution.
"""

import json
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CHUNKS_JSON  = "extracted/all_chunks.json"
CHROMA_DIR   = "./chroma_db"
COLLECTION   = "statics_8th_edition"
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
BATCH_SIZE   = 50

# ─── SETUP ────────────────────────────────────────────────────────────────────
print("Setting up ChromaDB...")
client   = chromadb.PersistentClient(path=CHROMA_DIR)
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

collection = client.get_or_create_collection(
    name=COLLECTION,
    embedding_function=embed_fn,
    metadata={"hnsw:space": "cosine"}
)
print(f"Collection '{COLLECTION}' ready")

# ─── LOAD ─────────────────────────────────────────────────────────────────────
with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

# ─── BUILD EMBED TEXT ─────────────────────────────────────────────────────────
def build_embed_text(chunk: dict) -> str:
    """
    The text that gets embedded — richer = better semantic search.
    Different templates for each source type.
    """
    pid    = chunk.get("problem_id", "")
    topic  = chunk.get("topic") or ""
    ch     = chunk.get("chapter") or ""
    source = chunk.get("source", "")
    text   = chunk.get("text", "").strip()

    base = f"""
Source: {source}
Chapter: {ch}
Topic: {topic}
Problem ID: {pid}
Content:
{text}
""".strip()

    if source == "question_bank":
        return f"{base}\nType: engineering mechanics statics question"
    elif source == "solution":
        return f"{base}\nType: worked numerical solution with equations and steps"
    elif source == "textbook":
        return f"{base}\nType: theory explanation, concepts, formulas, examples"

    return base


def build_metadata(chunk: dict) -> dict:
    """Flatten to primitives only (ChromaDB requirement)."""
    meta = {
        "problem_id": chunk.get("problem_id") or "unknown",
        "source":     chunk.get("source") or "unknown",
        "ch_num":     str(chunk.get("ch_num") or ""),
        "chapter":    chunk.get("chapter") or "",
        "topic":      chunk.get("topic") or "",
        "unit":       int(chunk.get("unit") or 0),
        "problem":    int(chunk.get("problem") or 0),
        "chunk_index": int(chunk.get("chunk_index") or 0),
        "chunk_type": chunk.get("chunk_type") or "",
        "chapter_topic_key": f"{chunk.get('chapter','')}_{chunk.get('topic','')}".strip("_"),
    }
    if chunk.get("page_count"):
        meta["page_count"] = int(chunk["page_count"])
    # Source-specific extras
    if chunk.get("image_path"):
        meta["image_path"] = chunk["image_path"]
    if chunk.get("solution_image_path"):
        meta["solution_image_path"] = chunk["solution_image_path"]
    if chunk.get("has_solution") is not None:
        meta["has_solution"] = str(chunk["has_solution"])
    if chunk.get("qb_file"):
        meta["qb_file"] = chunk["qb_file"]
    if chunk.get("page"):
        meta["page"] = int(chunk["page"])
    return meta


# ─── PREPARE ──────────────────────────────────────────────────────────────────
documents, metadatas, ids = [], [], []

for i, chunk in enumerate(chunks):
    text = build_embed_text(chunk)
    if not text.strip():
        continue

    source = chunk.get("source", "unknown")
    pid    = chunk.get("problem_id", f"idx{i}")
    doc_id = f"{source}_{pid}_{i}"

    documents.append(text)
    metadatas.append(build_metadata(chunk))
    ids.append(doc_id)

# ─── EMBED ────────────────────────────────────────────────────────────────────
print(f"\nEmbedding {len(documents)} documents...")
print(f"  Model      : {EMBED_MODEL}")
print(f"  Batch size : {BATCH_SIZE}")

from collections import Counter
src_counts = Counter(m["source"] for m in metadatas)
for src, cnt in src_counts.items():
    print(f"  {src:<20} {cnt} chunks")

for start in tqdm(range(0, len(documents), BATCH_SIZE), desc="Embedding"):
    end = min(start + BATCH_SIZE, len(documents))
    collection.upsert(
        documents=documents[start:end],
        metadatas=metadatas[start:end],
        ids=ids[start:end]
    )

print(f"\nDone! {collection.count()} documents in ChromaDB ({CHROMA_DIR})")
print("Next step: python 3_query.py")