"""Phase 2b - Index: turn chunks into embeddings and search them by meaning.

Usage:
    python -m repolens.index build  https://github.com/owner/repo
    python -m repolens.index search https://github.com/owner/repo "how are cookies stored"

An embedding is a list of numbers (here 384 of them) that captures what a piece of
text *means*. Texts with similar meaning get similar numbers, so "store cookies"
lands close to a function called `extract_cookies_to_jar` even though they share
almost no words. ChromaDB saves the embeddings on disk and finds the closest ones fast.
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Windows can't use Hugging Face's symlink cache; it still works, so hide the warning.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import chromadb
from fastembed import TextEmbedding

from repolens.chunk import chunk_files
from repolens.ingest import collect_files, resolve_source

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = DATA_DIR / "index"
MODELS_DIR = DATA_DIR / "models"

# Small and fast (67 MB), runs fine on a laptop CPU. It is a swappable setting:
# "jinaai/jina-embeddings-v2-base-code" is trained on code and worth comparing in Phase 5.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 256

_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    """Load the embedding model once (the first call downloads it)."""
    global _model
    if _model is None:
        _model = TextEmbedding(EMBEDDING_MODEL, cache_dir=str(MODELS_DIR))
    return _model


def collection_name(root: Path) -> str:
    # Chroma names: 3-63 characters, letters/digits/._- only.
    return re.sub(r"[^a-zA-Z0-9._-]", "_", root.name)[:63].strip("._-").ljust(3, "_")


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(INDEX_DIR))


def build_index(target: str) -> None:
    root = resolve_source(target)
    files, _ = collect_files(root)
    chunks = chunk_files(files)
    print(f"{len(files)} files -> {len(chunks)} chunks")

    client = get_client()
    name = collection_name(root)
    # Rebuild from scratch so the index never mixes old and new code.
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    # Cosine distance compares the *direction* of embeddings, the standard for text.
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    model = get_model()
    started = time.time()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        vectors = list(model.passage_embed([c.embedding_text() for c in batch]))
        collection.add(
            ids=[c.id for c in batch],
            embeddings=[v.tolist() for v in vectors],
            documents=[c.text for c in batch],
            metadatas=[{
                "path": c.path, "start_line": c.start_line, "end_line": c.end_line,
                "kind": c.kind, "name": c.name, "language": c.language,
            } for c in batch],
        )
        print(f"  embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}", end="\r")
    print(f"\nIndexed {len(chunks)} chunks in {time.time() - started:.1f}s -> collection '{name}'")


@dataclass
class SearchResult:
    path: str
    start_line: int
    end_line: int
    kind: str
    name: str
    text: str
    score: float  # 1.0 = identical meaning, around 0.5 = loosely related

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def search(target: str, query: str, k: int = 5) -> list[SearchResult]:
    root = resolve_source(target)
    collection = get_client().get_collection(collection_name(root))
    # query_embed (not passage_embed): this model encodes questions slightly
    # differently from the documents they should match.
    query_vector = next(iter(get_model().query_embed(query))).tolist()
    res = collection.query(query_embeddings=[query_vector], n_results=k)

    return [
        SearchResult(
            path=meta["path"], start_line=meta["start_line"], end_line=meta["end_line"],
            kind=meta["kind"], name=meta["name"], text=doc, score=1 - dist,
        )
        for meta, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0])
    ]


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("build", "search"):
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "build":
        build_index(sys.argv[2])
    else:
        query = " ".join(sys.argv[3:])
        for r in search(sys.argv[2], query):
            label = f"{r.kind} {r.name}".strip()
            print(f"\n[{r.score:.2f}] {r.citation}  ({label})")
            preview = "\n".join(r.text.splitlines()[:4])
            print("    " + preview.replace("\n", "\n    "))


if __name__ == "__main__":
    main()
