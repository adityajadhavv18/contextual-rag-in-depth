"""Index the chunks once per INDEX-TEXT variant: one Qdrant collection each, with the same three vectors.

  dense    bge-small "meaning" vector (384 numbers)           -> finds paraphrases
  bm25     sparse "keyword" vector (word id -> weight)        -> exact words: names, scores, grounds
  splade   learned sparse vector, adds related words           -> "hundred" also lights up "century"

A variant decides only what the three searches READ for each chunk (its index text). The payload text, which the LLM
reads and the harness matches evidence against, is always the bare chunk. So two collections hold the exact same
chunks and differ only in their labels, and any difference in scores comes from the label.

  plain          the chunk alone (r0, r1): the no-label control
  header         a free label: article title and heading path, then the chunk (r2a)
                   "Indian cricket team in Australia in 2020–21 › Test series › 1st Test › Day One"
  header_match   r2a's label plus the match box the chunk belongs to (r2b)
                   "17–21 December 2020 (D/N) | India v Australia | Adelaide Oval, Adelaide"
                 Both labels are built by code from the article's own structure (chunks.py): no LLM, no API.
  header_match_note  r2b's label plus an LLM note on what happens in the chunk (notes.py, section mode) (r3)

Batches are small (8) and points upload from a generator because this runs on an 8 GB laptop. Model outputs are
cached by text hash (data/cricket/cache/), so a variant only embeds index texts it hasn't seen before.

Run from contextual_rag_in_depth/ (Qdrant must be running: open -a Docker; docker start qdrant):
    uv run cricket/ingest.py plain
    uv run cricket/ingest.py header header_match
"""

import hashlib
import pickle
import statistics
import sys
import time
import uuid
from functools import cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.common.utils import remove_non_alphanumeric
from qdrant_client import QdrantClient, models

from chunks import DATA, EMBED_MODEL, Chunk, load_chunks

QDRANT_URL = "http://localhost:6333"
BM25_MODEL = "Qdrant/bm25"
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
DENSE, BM25, SPLADE = "dense", "bm25", "splade"
CACHE = DATA / "cache"

@cache
def section_notes() -> dict:
    from notes import load_notes  # notes.py imports this module, so import it only when a variant needs it
    return load_notes("section")


VARIANTS = {  # name -> the index text of a chunk
    "plain": lambda c: c.text,
    "header": lambda c: f"{' › '.join(c.path)}\n{c.text}",
    "header_match": lambda c: "\n".join([" › ".join(c.path), *c.match, c.text]),
    "header_match_note": lambda c: "\n".join([" › ".join(c.path), *c.match, section_notes()[c.chunk_id]["note"],
                                              c.text]),
}


def collection(variant: str) -> str:
    return f"cricket_{variant}"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))  # Qdrant ids must be ints or UUIDs


def avg_words(texts: list[str]) -> float:
    """BM25 discounts words in long chunks relative to the AVERAGE length (FastEmbed assumes 256); measure ours."""
    probe = SparseTextEmbedding(BM25_MODEL).model
    return statistics.mean(len(probe._stem(probe.tokenizer.tokenize(remove_non_alphanumeric(t)))) for t in texts)


def cached(tag: str, texts: list[str], embed) -> list:
    """embed(list_of_texts) -> outputs, cached per text under data/cricket/cache/<tag>.pkl."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag.replace('/', '_')}.pkl"
    store = pickle.loads(path.read_bytes()) if path.exists() else {}
    keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
    todo = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in store))
    if todo:
        start = time.perf_counter()
        for n, (i, out) in enumerate(zip(todo, embed([texts[i] for i in todo])), start=1):
            store[keys[i]] = out
            if n % 500 == 0:
                print(f"     {tag}: {n}/{len(todo)} ({time.perf_counter() - start:.0f}s)", flush=True)
        path.write_bytes(pickle.dumps(store))
        print(f"     embedded {len(todo)} texts for {tag} in {time.perf_counter() - start:.0f}s", flush=True)
    return [store[k] for k in keys]


def sparse(v) -> models.SparseVector:
    return models.SparseVector(indices=v.indices.tolist(), values=v.values.tolist())


def ingest(variant: str, chunks: list[Chunk] | None = None, extra_payload: dict[str, dict] | None = None) -> None:
    chunks = chunks or load_chunks()
    texts = [VARIANTS[variant](c) for c in chunks]
    name = collection(variant)
    print(f"1. {len(chunks)} chunks from {len({c.doc_id for c in chunks})} articles -> '{name}'", flush=True)

    dense_model = TextEmbedding(EMBED_MODEL)
    vectors = {
        DENSE: cached(f"dense_{EMBED_MODEL}", texts, lambda ts: dense_model.passage_embed(ts, batch_size=8)),
        BM25: list(SparseTextEmbedding(BM25_MODEL, avg_len=avg_words(texts)).embed(texts)),  # fast: no cache
        SPLADE: cached(f"splade_{SPLADE_MODEL}", texts,
                       lambda ts: SparseTextEmbedding(SPLADE_MODEL).embed(ts, batch_size=8)),
    }
    print(f"2. vectors: {', '.join(vectors)}", flush=True)

    client = QdrantClient(url=QDRANT_URL, timeout=300)
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=len(vectors[DENSE][0]), distance=models.Distance.COSINE)},
        # BM25 vectors hold only the term-frequency half of the formula; Modifier.IDF makes Qdrant add the other
        # half (how RARE each word is across all chunks) at query time. SPLADE weights are complete already.
        sparse_vectors_config={BM25: models.SparseVectorParams(modifier=models.Modifier.IDF),
                               SPLADE: models.SparseVectorParams()},
    )
    client.upload_points(
        name,
        points=(models.PointStruct(
            id=point_id(c.chunk_id),
            vector={DENSE: vectors[DENSE][i].tolist(), BM25: sparse(vectors[BM25][i]),
                    SPLADE: sparse(vectors[SPLADE][i])},
            payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "path": list(c.path), "text": c.text,
                     **(extra_payload or {}).get(c.chunk_id, {})})
            for i, c in enumerate(chunks)),
        batch_size=32,
        wait=True,
    )
    print(f"3. stored: {client.count(name, exact=True).count} points in '{name}'", flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:] or ["plain"]:
        ingest(v)
