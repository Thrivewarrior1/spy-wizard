"""Semantic embedding search for Spy Wizard.

Why this exists
---------------
Keyword/tag search — even with a controlled vocabulary and a strict
Gemini judge — fundamentally requires the query token to EXACT-MATCH a
token on the product. A product tagged `over-the-knee-boots` misses a
`knee-high boots` query; a German store's `Kniehohe Stiefel` misses an
English query entirely. And a full-catalog vision backfill to produce
those tags takes hours.

Embeddings solve all three at once:
  - MULTILINGUAL: gemini-embedding-001 maps "knee-high boots",
    "Kniehohe Stiefel", "bottes hautes", "botas altas" to nearly the
    same point in vector space. No translation step.
  - SEMANTIC: "summer dress" retrieves "light floral sundress" with
    zero shared keywords.
  - FAST BACKFILL: embeddings are text (not images) and batch 100 per
    API call. The whole ~5000-product catalog embeds in ~2-3 minutes
    for ~2 cents — vs. hours for per-image vision.

Architecture
------------
  1. Every product gets an embedding vector (this module + the
     /api/admin/backfill-embeddings endpoint). Source text is
     title + vision_description + cleaned tags.
  2. At query time: embed the query, cosine-similarity rank all
     products, take the top-N candidates. This is the RECALL stage —
     it surfaces the right products regardless of language.
  3. The existing Gemini strict judge runs on the top candidates for
     the final 100%-precision PRECISION cut (drops the ankle boots
     that ranked near the knee-high boots).

The in-memory index
-------------------
5000 products × 768 float32 = ~15 MB — trivially small to hold in
process memory. We load it lazily on first search after a deploy,
cache it in a module global, and invalidate it when a backfill runs.
Cosine similarity over the whole matrix is a single numpy matvec
(~1 ms).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import struct
import threading
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

try:
    import numpy as np  # type: ignore
    _NP_OK = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore
    _NP_OK = False


EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
# SEMANTIC_SIMILARITY tested cleanest for our query-vs-product setup
# (English query scored German/French products ~0.91, off-category
# dresses ~0.78). RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT is the alternative
# asymmetric pair — switch via env if recall needs tuning.
EMBED_TASK = os.getenv("EMBED_TASK", "SEMANTIC_SIMILARITY")
EMBED_BATCH = 100          # gemini batchEmbedContents accepts up to 100
EMBED_TIMEOUT = 30.0


def _embed_url(batch: bool = False) -> str:
    verb = "batchEmbedContents" if batch else "embedContent"
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{EMBED_MODEL}:{verb}"
    )


# ---------------------------------------------------------------------
# Embedding API calls.
# ---------------------------------------------------------------------
async def embed_query(text: str) -> Optional[list[float]]:
    """Embed a single query string. Returns the float vector or None
    on failure."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not (text or "").strip():
        return None
    payload = {
        "content": {"parts": [{"text": text.strip()}]},
        "taskType": EMBED_TASK,
        "outputDimensionality": EMBED_DIM,
    }
    try:
        async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
            r = await client.post(
                _embed_url(batch=False),
                params={"key": api_key},
                json=payload,
            )
        if r.status_code != 200:
            logger.warning("embed_query HTTP %d: %s", r.status_code, r.text[:200])
            return None
        return r.json()["embedding"]["values"]
    except Exception as e:
        logger.warning("embed_query exception: %s", e)
        return None


async def embed_texts(texts: list[str]) -> list[Optional[list[float]]]:
    """Embed a list of texts via batchEmbedContents (100 per call).
    Returns a list of vectors aligned with the input; entries are None
    where embedding failed. Never raises."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return [None] * len(texts)
    out: list[Optional[list[float]]] = []
    async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
        for start in range(0, len(texts), EMBED_BATCH):
            chunk = texts[start:start + EMBED_BATCH]
            requests = [
                {
                    "model": f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": (t or " ").strip() or " "}]},
                    "taskType": EMBED_TASK,
                    "outputDimensionality": EMBED_DIM,
                }
                for t in chunk
            ]
            try:
                r = await client.post(
                    _embed_url(batch=True),
                    params={"key": api_key},
                    json={"requests": requests},
                )
                if r.status_code != 200:
                    logger.warning(
                        "embed_texts batch HTTP %d: %s",
                        r.status_code, r.text[:200],
                    )
                    out.extend([None] * len(chunk))
                    continue
                data = r.json()
                embs = data.get("embeddings", [])
                for i in range(len(chunk)):
                    if i < len(embs) and "values" in embs[i]:
                        out.append(embs[i]["values"])
                    else:
                        out.append(None)
            except Exception as e:
                logger.warning("embed_texts batch exception: %s", e)
                out.extend([None] * len(chunk))
    return out


# ---------------------------------------------------------------------
# Product embedding text builder.
# ---------------------------------------------------------------------
def build_embedding_text(
    *, title: str, vision_description: str, ai_tags: str,
) -> str:
    """Compose the natural-language text we embed for a product.

    Priority order (most reliable signal first):
      1. vision_description — ground truth from the image
      2. title — the merchant's own words (any language)
      3. cleaned tags — the `img:` prefixes stripped so the embedder
         sees natural words ("knee-high boots" not "img:type:knee-high-boots")
    """
    parts: list[str] = []
    t = (title or "").strip()
    if t:
        parts.append(t)
    vd = (vision_description or "").strip()
    if vd:
        parts.append(vd)
    tags = _clean_tags_for_embedding(ai_tags or "")
    if tags:
        parts.append(tags)
    return ". ".join(parts)[:2000]


def _clean_tags_for_embedding(ai_tags: str) -> str:
    """Turn the raw ai_tags string (which may include img:type:foo,
    img:attr:bar || text tags) into a natural comma list of the
    values only, deduped."""
    if not ai_tags:
        return ""
    seen: set[str] = set()
    out: list[str] = []
    # Split on both the "||" separator and commas.
    for chunk in ai_tags.replace("||", ",").split(","):
        v = chunk.strip()
        if not v:
            continue
        # Strip img: namespaces → keep the value after the last colon.
        if v.startswith("img:"):
            v = v.split(":")[-1]
        v = v.replace("-", " ").strip().lower()
        if v and v not in seen and len(v) > 1:
            seen.add(v)
            out.append(v)
    return ", ".join(out)


# ---------------------------------------------------------------------
# In-memory similarity index.
# ---------------------------------------------------------------------
_INDEX_LOCK = threading.Lock()
_INDEX: dict = {"ids": None, "matrix": None, "loaded": False, "count": 0}


def invalidate_index() -> None:
    """Mark the in-memory index stale so the next search reloads it.
    Called after a backfill writes new embeddings."""
    with _INDEX_LOCK:
        _INDEX["loaded"] = False


def encode_vector(vec: list[float]) -> str:
    """Compact storage: pack floats as little-endian float32 bytes and
    base64-encode. ~4KB per 768-dim vector vs ~10KB for JSON, and
    decoding via frombuffer avoids building a huge list of Python
    float objects (the memory spike that can OOM a 512MB instance
    when loading thousands of vectors). Prefixed 'b64:' so the loader
    can distinguish it from legacy JSON storage."""
    buf = struct.pack("<%df" % len(vec), *[float(x) for x in vec])
    return "b64:" + base64.b64encode(buf).decode("ascii")


def _parse_vector(raw: str):
    """Decode a stored embedding. Handles the compact 'b64:' float32
    format AND legacy JSON arrays. Returns a list of floats or None.
    Never raises."""
    if not raw:
        return None
    try:
        if raw.startswith("b64:"):
            data = base64.b64decode(raw[4:])
            n = len(data) // 4
            return list(struct.unpack("<%df" % n, data))
        v = json.loads(raw)
        if isinstance(v, list) and v:
            return v
    except Exception:
        pass
    return None


_INDEX_MAX = int(os.getenv("EMBED_INDEX_MAX", "12000"))


def load_index(db) -> int:
    """(Re)load product embeddings into an in-memory numpy matrix.
    Returns the number of vectors loaded. Exception-safe — on ANY
    error it leaves the index empty and marks it loaded so a failing
    load can never crash-loop a request or the health check.

    Memory-conscious: streams rows in chunks and decodes each vector
    straight into a pre-allocated float32 matrix via frombuffer,
    avoiding a giant intermediate list of Python float objects (the
    spike that OOMs a 512MB instance when thousands of vectors load
    at once). Capped at EMBED_INDEX_MAX vectors.
    """
    if not _NP_OK:
        with _INDEX_LOCK:
            _INDEX["loaded"] = True
        return 0
    try:
        from models import Product
        rows = (
            db.query(Product.id, Product.embedding)
            .filter(Product.embedding.isnot(None))
            .order_by(Product.last_scraped.desc())
            .limit(_INDEX_MAX)
            .all()
        )
        ids: list[int] = []
        mats: list = []  # list of small (1,dim) float32 arrays
        for pid, emb in rows:
            v = _parse_vector(emb) if emb else None
            if not v:
                continue
            arr = np.asarray(v, dtype=np.float32)
            if arr.size == 0:
                continue
            ids.append(int(pid))
            mats.append(arr)
        with _INDEX_LOCK:
            if ids:
                mat = np.vstack(mats).astype(np.float32, copy=False)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                mat = mat / norms
                _INDEX["ids"] = np.asarray(ids, dtype=np.int64)
                _INDEX["matrix"] = mat
                _INDEX["count"] = len(ids)
            else:
                _INDEX["ids"] = None
                _INDEX["matrix"] = None
                _INDEX["count"] = 0
            _INDEX["loaded"] = True
        logger.info("embeddings: loaded index with %d vectors", len(ids))
        return len(ids)
    except Exception as e:
        logger.warning("embeddings: load_index failed (%s) — empty index", e)
        with _INDEX_LOCK:
            _INDEX["ids"] = None
            _INDEX["matrix"] = None
            _INDEX["count"] = 0
            _INDEX["loaded"] = True
        return 0


def ensure_index(db) -> int:
    """Load the index if not already loaded. Returns vector count."""
    with _INDEX_LOCK:
        if _INDEX["loaded"]:
            return _INDEX["count"]
    return load_index(db)


def rank_ids_by_query(
    db,
    query_vec: list[float],
    *,
    scoped_ids: Optional[set] = None,
    top_k: int = 80,
) -> list[tuple[int, float]]:
    """Return the top_k (product_id, cosine_similarity) pairs for the
    query vector, restricted to scoped_ids when provided.

    Cosine similarity is a normalised dot product; the index matrix is
    pre-normalised at load time so this is a single matvec.
    """
    if not _NP_OK or query_vec is None:
        return []
    ensure_index(db)
    with _INDEX_LOCK:
        ids = _INDEX["ids"]
        mat = _INDEX["matrix"]
        if ids is None or mat is None or _INDEX["count"] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn
        sims = mat @ q  # (N,) cosine similarities
        # Restrict to scoped ids if provided.
        if scoped_ids is not None:
            mask = np.fromiter(
                (int(i) in scoped_ids for i in ids),
                dtype=bool, count=len(ids),
            )
            if not mask.any():
                return []
            idxs = np.nonzero(mask)[0]
            sub_sims = sims[idxs]
            order = np.argsort(-sub_sims)[:top_k]
            return [(int(ids[idxs[o]]), float(sub_sims[o])) for o in order]
        order = np.argsort(-sims)[:top_k]
        return [(int(ids[o]), float(sims[o])) for o in order]


def index_stats() -> dict:
    with _INDEX_LOCK:
        return {
            "loaded": _INDEX["loaded"],
            "count": _INDEX["count"],
            "numpy_available": _NP_OK,
            "model": EMBED_MODEL,
            "dim": EMBED_DIM,
            "task": EMBED_TASK,
        }
