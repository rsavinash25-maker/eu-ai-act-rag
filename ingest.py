"""Load documents, chunk them, embed them, store them in Chroma.

Usage:
    python ingest.py                       # uses CHUNK_SIZE / CHUNK_OVERLAP from config
    python ingest.py --chunk-size 300 --overlap 30 --rebuild
"""
import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path

import tiktoken

import config

ENC = tiktoken.get_encoding("cl100k_base")  # used only to count tokens consistently


def n_tokens(text: str) -> int:
    return len(ENC.encode(text))


# ---------------------------------------------------------------- loading
def load_documents(docs_dir: str | None = None) -> list[dict]:
    """Returns [{source, pages: [(page_number, text), ...]}]. Non-PDFs are one 'page'."""
    docs_dir = docs_dir or config.DOCS_DIR
    docs = []
    for path in sorted(Path(docs_dir).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            pages = [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
        elif suffix in {".txt", ".md"}:
            pages = [(1, path.read_text(encoding="utf-8", errors="ignore"))]
        else:
            continue
        header = ""
        # Files from split_regulation.py start with a "# ..." context header.
        if suffix == ".md" and pages[0][1].startswith("# "):
            header, _, body = pages[0][1].partition("\n\n")
            pages = [(1, body)]
        pages = [(n, t) for n, t in pages if t.strip()]
        if pages:
            docs.append({"source": path.name, "pages": pages, "header": header.strip()})
    return docs


# ---------------------------------------------------------------- chunking
@dataclass
class Unit:
    """A sentence (or a hard-split fragment of a very long one)."""
    text: str
    page: int
    para_start: bool
    ntok: int


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def to_units(pages: list[tuple[int, str]], max_unit: int) -> list[Unit]:
    units = []
    for page, text in pages:
        for para in re.split(r"\n\s*\n", text):
            para = re.sub(r"\s+", " ", para).strip()
            if not para:
                continue
            first = True
            for sent in SENT_SPLIT.split(para):
                toks = ENC.encode(sent)
                # Very long "sentences" (tables, lists without punctuation) get hard-split.
                for i in range(0, len(toks), max_unit):
                    piece = ENC.decode(toks[i : i + max_unit])
                    units.append(Unit(piece, page, first, len(toks[i : i + max_unit])))
                    first = False
    return units


def pack(units: list[Unit], size: int, overlap: int) -> list[list[Unit]]:
    """Greedily pack whole sentences into chunks of <= size tokens.
    Overlap is made of whole trailing sentences totalling <= overlap tokens."""
    chunks, cur, cur_tok = [], [], 0
    for u in units:
        if cur and cur_tok + u.ntok > size:
            chunks.append(cur)
            carry, t = [], 0
            for v in reversed(cur):
                if t + v.ntok > overlap:
                    break
                carry.insert(0, v)
                t += v.ntok
            cur, cur_tok = carry, t
            if cur_tok + u.ntok > size:  # carry + next unit won't fit: drop the carry
                cur, cur_tok = [], 0
        cur.append(u)
        cur_tok += u.ntok
    if cur:
        chunks.append(cur)
    return chunks


def join_units(units: list[Unit]) -> str:
    out = ""
    for i, u in enumerate(units):
        sep = "" if i == 0 else ("\n\n" if u.para_start else " ")
        out += sep + u.text
    return out


def chunk_document(doc: dict, size: int, overlap: int) -> list[dict]:
    header = doc.get("header", "")
    # The header is repeated on every chunk, so it comes out of the token budget.
    budget = max(size - n_tokens(header), 50) if header else size
    units = to_units(doc["pages"], max_unit=budget)
    chunks = []
    for i, group in enumerate(pack(units, budget, overlap)):
        text = join_units(group)
        if header:
            text = header + "\n\n" + text
        chunks.append(
            {
                "id": f"{doc['source']}::{i}",
                "text": text,
                "metadata": {
                    "source": doc["source"],
                    "title": header.splitlines()[0].lstrip("# ") if header else "",
                    "page_start": group[0].page,
                    "page_end": group[-1].page,
                    "chunk_index": i,
                    "n_tokens": n_tokens(text),
                },
            }
        )
    return chunks


# ---------------------------------------------------------------- indexing
def get_client():
    import chromadb

    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def build_index(chunk_size: int, overlap: int, rebuild: bool = False, batch: int = 64,
                docs_dir: str | None = None, tag: str = "") -> dict:
    from embeddings import Embedder

    client = get_client()
    name = config.collection_name(chunk_size, overlap, tag)
    existing = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if name in existing:
        if not rebuild:
            col = client.get_collection(name)
            metas = col.get(include=["metadatas"])["metadatas"]
            print(f"Collection {name} exists ({len(metas)} chunks). Use --rebuild to redo.")
            return {
                "collection": name,
                "documents": len({m["source"] for m in metas}),
                "chunks": len(metas),
                "avg_chunk_tokens": round(sum(m["n_tokens"] for m in metas) / max(len(metas), 1)),
                "index_seconds": "cached",
            }
        client.delete_collection(name)

    col = client.create_collection(
        name,
        metadata={
            "hnsw:space": "cosine",
            "chunk_size": chunk_size,
            "overlap": overlap,
            "embed_model": config.EMBED_MODEL,
            "docs_dir": docs_dir or config.DOCS_DIR,
        },
    )

    t0 = time.perf_counter()
    docs = load_documents(docs_dir)
    if not docs:
        raise SystemExit(f"No .pdf/.txt/.md files found in {docs_dir or config.DOCS_DIR}")
    chunks = [c for d in docs for c in chunk_document(d, chunk_size, overlap)]

    embedder = Embedder()
    if embedder.max_tokens and chunk_size > embedder.max_tokens:
        print(
            f"WARNING: chunk size {chunk_size} exceeds {config.EMBED_MODEL}'s limit of "
            f"{embedder.max_tokens} tokens; the end of each chunk won't be embedded."
        )

    for i in range(0, len(chunks), batch):
        b = chunks[i : i + batch]
        col.add(
            ids=[c["id"] for c in b],
            documents=[c["text"] for c in b],
            metadatas=[c["metadata"] for c in b],
            embeddings=embedder.embed_documents([c["text"] for c in b]),
        )
        print(f"  embedded {min(i + batch, len(chunks))}/{len(chunks)}", end="\r")

    sizes = [c["metadata"]["n_tokens"] for c in chunks]
    stats = {
        "collection": name,
        "documents": len(docs),
        "chunks": len(chunks),
        "avg_chunk_tokens": round(sum(sizes) / len(sizes)),
        "index_seconds": round(time.perf_counter() - t0, 1),
    }
    print("\n", stats)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--docs-dir", default=None, help="default: config.DOCS_DIR")
    ap.add_argument("--tag", default="", help='e.g. "struct" when indexing data/docs_structured')
    a = ap.parse_args()
    build_index(a.chunk_size, a.overlap, a.rebuild, docs_dir=a.docs_dir, tag=a.tag)
