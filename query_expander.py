"""AI-powered query expansion + re-ranking for Spy Wizard search.

Why this exists
---------------
The pre-expander search treated "prom dress" as two whitespace-separated
ASCII tokens and required BOTH to match somewhere in the catalog. "Prom"
isn't a registered category, isn't in any of the multilingual TRANSLATIONS
dicts, and isn't in the Gemini classifier's prompt vocabulary — so a
catalog containing 26 prom-suitable evening gowns surfaced 2 results
(the only products with literal "prom" in their title). Same story for
"wedding guest dress", "festival outfit", "office workwear", etc.

The expander fixes this by routing every user query through Gemini
Flash once, producing:
  * canonical English synonyms ("prom dress" -> evening gown / formal
    dress / ball gown / cocktail dress / gala dress / homecoming dress)
  * occasion / style / material / color tags from the same controlled
    vocabularies the classifier uses (so they line up with the new
    ai_tags emitted by classifier.py)
  * multilingual noun translations (Abendkleid, robe de soiree,
    vestido de gala, abito da sera, jurk, sukienka)
  * 3-5 short phrases a product title might actually use

The expansion is then fed to a hybrid scorer (main.py:hybrid_search)
that ranks every candidate product by how many tokens overlap across
title / ai_tags / product_category / subniche. Finally the top 50 are
re-ranked by another Gemini call that sees the user query + candidate
titles together and discards anything that's clearly off-topic.

Caching
-------
- In-process LRU keyed by SHA256(normalised query). Default 1024 entries.
  Same query string returns the same expansion forever (until process
  restart) — costs one Gemini call the first time, zero thereafter.
- Re-rank cache keyed by (query_hash, sorted_candidate_ids_hash). Tiny
  LRU because the candidate set varies per scrape.

Fallback
--------
- Missing GEMINI_API_KEY -> expander returns a regex-style identity
  expansion (the original tokens only) and the re-rank step is a no-op.
  Search degrades gracefully to the old behaviour rather than throwing.
- Gemini failure mid-request -> same identity fallback for that query.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# Same model + endpoint helpers as classifier.py. Kept local so this
# module is import-safe even before classifier loads its full prompt.
_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-flash-latest"]


def _gemini_url(model: str) -> str:
    return (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )


# Vocabularies — MUST stay in sync with the classifier.py prompt so that
# tags Gemini emits match tags this expander asks for. If you add a
# tag here, add it to the classifier prompt's vocabulary block too.
_OCCASION_TAGS = {
    "prom", "wedding", "wedding-guest", "bridesmaid", "mother-of-bride",
    "cocktail", "evening", "gala", "formal", "ball", "party", "festival",
    "casual", "work", "office", "business", "beach", "swim", "vacation",
    "lounge", "sleep", "gym", "athletic", "hiking", "ski", "oktoberfest",
    "graduation", "dance", "club", "brunch", "date-night", "holiday",
}
_STYLE_TAGS = {
    "maxi", "midi", "mini", "bodycon", "wrap", "a-line", "sheath", "shift",
    "ball-gown", "slip", "kaftan", "jumpsuit", "romper", "two-piece",
    "crop-top", "oversized", "fitted", "tailored", "relaxed",
    "high-waisted", "low-rise", "pleated", "ruched", "ruffled",
    "asymmetric", "backless", "cut-out", "off-the-shoulder", "peplum",
}
_MATERIAL_TAGS = {
    "silk", "satin", "lace", "chiffon", "sequin", "velvet", "tulle",
    "cotton", "linen", "denim", "leather", "suede", "wool", "cashmere",
    "knit", "mesh", "crochet", "fleece", "polyester", "viscose", "rayon",
    "organza", "jersey", "ribbed", "faux-fur", "faux-leather",
}
_COLOR_TAGS = {
    "black", "white", "red", "blue", "navy", "green", "pink", "beige",
    "ivory", "champagne", "gold", "silver", "brown", "cream", "burgundy",
    "emerald", "floral", "striped", "animal-print", "metallic", "pastel",
    "neutral",
}


@dataclass
class ExpansionResult:
    """Structured query expansion. Empty lists are valid (means
    "expander couldn't add anything"); the scorer treats them as
    a no-op rather than an error."""
    original: str
    canonical_terms: list[str] = field(default_factory=list)
    occasion_tags: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    material_tags: list[str] = field(default_factory=list)
    color_tags: list[str] = field(default_factory=list)
    multilingual_nouns: list[str] = field(default_factory=list)
    semantic_phrases: list[str] = field(default_factory=list)
    cached: bool = False
    expander_used: str = "none"  # 'gemini' | 'fallback' | 'cache'

    def all_terms(self) -> set[str]:
        """Every token / phrase the expansion produced, lowercased.
        Used by the scorer for a fast Python-side in-check after the
        SQL prefilter has narrowed the candidate set."""
        out: set[str] = set()
        for lst in (
            self.canonical_terms, self.occasion_tags, self.style_tags,
            self.material_tags, self.color_tags, self.multilingual_nouns,
            self.semantic_phrases,
        ):
            for t in lst:
                if t and isinstance(t, str):
                    out.add(t.strip().lower())
        out.update(self.original.lower().split())
        out.discard("")
        return out

    def tag_terms(self) -> set[str]:
        """Tags only — the subset the classifier might have emitted.
        Used to score ai_tags matches more heavily than free-text
        matches."""
        out: set[str] = set()
        for lst in (
            self.occasion_tags, self.style_tags, self.material_tags,
            self.color_tags,
        ):
            for t in lst:
                if t and isinstance(t, str):
                    out.add(t.strip().lower())
        return out


# ---------------------------------------------------------------------
# Cache layer.
# ---------------------------------------------------------------------
_EXPANSION_CACHE: dict[str, ExpansionResult] = {}
_EXPANSION_CACHE_ORDER: list[str] = []
_EXPANSION_CACHE_MAX = 1024


def _query_hash(q: str) -> str:
    return hashlib.sha256(q.strip().lower().encode("utf-8")).hexdigest()


def _cache_get(qh: str) -> Optional[ExpansionResult]:
    res = _EXPANSION_CACHE.get(qh)
    if res is not None:
        return res
    return None


def _cache_put(qh: str, exp: ExpansionResult) -> None:
    _EXPANSION_CACHE[qh] = exp
    _EXPANSION_CACHE_ORDER.append(qh)
    while len(_EXPANSION_CACHE_ORDER) > _EXPANSION_CACHE_MAX:
        old = _EXPANSION_CACHE_ORDER.pop(0)
        _EXPANSION_CACHE.pop(old, None)


# ---------------------------------------------------------------------
# Identity / regex fallback (no Gemini available).
# ---------------------------------------------------------------------
def _identity_expansion(q: str) -> ExpansionResult:
    """When Gemini is unavailable or fails, return JUST the original
    tokens. The scorer will still find direct keyword matches; this is
    strictly worse than a Gemini expansion but never returns zero
    results when the old search would have found something."""
    return ExpansionResult(
        original=q,
        canonical_terms=[w for w in re.split(r"\s+", q.lower()) if w],
        expander_used="fallback",
    )


# ---------------------------------------------------------------------
# Gemini call.
# ---------------------------------------------------------------------
_EXPAND_PROMPT_TEMPLATE = """You are the query expander for a multilingual fashion + general product search.

The user typed: {query!r}

The catalog spans English / German / French / Italian / Spanish / Dutch / Polish stores. Tags on each product are drawn from these CONTROLLED VOCABULARIES (use the EXACT same tokens — never invent variants):

OCCASION: prom, wedding, wedding-guest, bridesmaid, mother-of-bride, cocktail, evening, gala, formal, ball, party, festival, casual, work, office, business, beach, swim, vacation, lounge, sleep, gym, athletic, hiking, ski, oktoberfest, graduation, dance, club, brunch, date-night, holiday

STYLE: maxi, midi, mini, bodycon, wrap, a-line, sheath, shift, ball-gown, slip, kaftan, jumpsuit, romper, two-piece, crop-top, oversized, fitted, tailored, relaxed, high-waisted, low-rise, pleated, ruched, ruffled, asymmetric, backless, cut-out, off-the-shoulder, peplum

MATERIAL: silk, satin, lace, chiffon, sequin, velvet, tulle, cotton, linen, denim, leather, suede, wool, cashmere, knit, mesh, crochet, fleece, polyester, viscose, rayon, organza, jersey, ribbed, faux-fur, faux-leather

COLOR: black, white, red, blue, navy, green, pink, beige, ivory, champagne, gold, silver, brown, cream, burgundy, emerald, floral, striped, animal-print, metallic, pastel, neutral

Produce a JSON expansion of the query. Be GENEROUS — better to over-expand than under-expand. The goal is for a shopper typing this query to see EVERY product across the catalog that a competent human would consider relevant.

Return JSON:
{{
  "canonical_terms": ["..."],         // 5-15 English synonyms / near-synonyms of the query
  "occasion_tags":   ["..."],         // tokens from OCCASION above that apply
  "style_tags":      ["..."],         // tokens from STYLE above
  "material_tags":   ["..."],         // tokens from MATERIAL above
  "color_tags":      ["..."],         // tokens from COLOR above (often empty unless the query names a color)
  "multilingual_nouns": ["..."],      // 6-15 nouns covering the same concept in German, French, Italian, Spanish, Dutch, Polish (e.g. Abendkleid, Robe de soiree, Vestido de gala, Abito da sera, Jurk, Sukienka)
  "semantic_phrases": ["..."]         // 3-5 short phrases a product title would actually use (e.g. "evening gown", "formal dress", "floor-length gown")
}}

Only return tokens that GENUINELY apply. If the query is "wedding guest dress" the occasion_tags include wedding-guest AND wedding AND cocktail AND formal AND evening — all four are plausible. If the query is "athletic shorts" the material_tags are probably empty and the occasion_tags reduce to athletic, gym, casual.

For non-fashion queries (e.g. "smartwatch", "table lamp", "blood pressure monitor"), the apparel vocabularies are mostly empty; lean on canonical_terms + multilingual_nouns + semantic_phrases. Output the JSON only, no markdown fences or commentary."""


async def _call_gemini_expand(q: str, api_key: str) -> Optional[dict]:
    """Single Gemini call, no retry. Returns the parsed JSON or None."""
    prompt = _EXPAND_PROMPT_TEMPLATE.format(query=q)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    for model in [_MODEL, *_FALLBACK_MODELS]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(
                    _gemini_url(model),
                    params={"key": api_key},
                    json=payload,
                )
            if r.status_code != 200:
                logger.warning(
                    "expand_query: %s returned HTTP %d: %s",
                    model, r.status_code, r.text[:200],
                )
                continue
            data = r.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                logger.warning("expand_query: unexpected response shape from %s", model)
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Strip markdown fences if Gemini ignored the prompt
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    logger.warning(
                        "expand_query: bad JSON from %s: %s", model, text[:200],
                    )
                    continue
            return parsed
        except Exception as e:
            logger.warning("expand_query: %s exception %s", model, e)
            continue
    return None


def _normalise_list(raw, allowed: Optional[set[str]] = None, cap: int = 30) -> list[str]:
    """Normalise a Gemini-returned list field: strip, lowercase, dedup,
    cap length, optionally restrict to a controlled vocabulary."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw[:cap]:
        if not isinstance(item, str):
            continue
        v = item.strip().lower()
        if not v or v in seen:
            continue
        if allowed is not None and v not in allowed:
            continue
        seen.add(v)
        out.append(v)
    return out


async def expand_query(q: str) -> ExpansionResult:
    """Public entry point. Returns an ExpansionResult.

    Cache hits are O(1); cache misses cost one Gemini call (~500ms-2s).
    """
    qn = q.strip()
    if not qn:
        return ExpansionResult(original=q)

    qh = _query_hash(qn)
    cached = _cache_get(qh)
    if cached is not None:
        # Don't mutate the cached object — return a copy with cached=True
        return ExpansionResult(
            original=cached.original,
            canonical_terms=list(cached.canonical_terms),
            occasion_tags=list(cached.occasion_tags),
            style_tags=list(cached.style_tags),
            material_tags=list(cached.material_tags),
            color_tags=list(cached.color_tags),
            multilingual_nouns=list(cached.multilingual_nouns),
            semantic_phrases=list(cached.semantic_phrases),
            cached=True,
            expander_used=cached.expander_used,
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        exp = _identity_expansion(qn)
        _cache_put(qh, exp)
        return exp

    parsed = await _call_gemini_expand(qn, api_key)
    if parsed is None:
        exp = _identity_expansion(qn)
        # Don't cache fallback expansions — next call might succeed
        return exp

    exp = ExpansionResult(
        original=qn,
        canonical_terms=_normalise_list(parsed.get("canonical_terms"), cap=20),
        occasion_tags=_normalise_list(parsed.get("occasion_tags"), allowed=_OCCASION_TAGS),
        style_tags=_normalise_list(parsed.get("style_tags"), allowed=_STYLE_TAGS),
        material_tags=_normalise_list(parsed.get("material_tags"), allowed=_MATERIAL_TAGS),
        color_tags=_normalise_list(parsed.get("color_tags"), allowed=_COLOR_TAGS),
        multilingual_nouns=_normalise_list(parsed.get("multilingual_nouns"), cap=25),
        semantic_phrases=_normalise_list(parsed.get("semantic_phrases"), cap=10),
        expander_used="gemini",
    )
    _cache_put(qh, exp)
    return exp


# ---------------------------------------------------------------------
# Re-rank top N with Gemini.
# ---------------------------------------------------------------------
_RERANK_CACHE: dict[tuple, list[dict]] = {}
_RERANK_CACHE_ORDER: list[tuple] = []
_RERANK_CACHE_MAX = 256


_RERANK_PROMPT_TEMPLATE = """You are re-ranking product search results.

The user searched: {query!r}

Below are {n} candidate products from a multilingual catalog. For EACH product, decide how well it satisfies the user's intent and whether it should be dropped from the results.

Scoring rubric (0-100):
  100  perfect match — a shopper looking for this exact thing would buy this product
   80  strong match — clearly relevant
   60  reasonable match — relevant but not the best example
   40  weak match — tangentially relevant
   20  poor match — only loosely related
    0  irrelevant — would frustrate the user

drop=true for anything with score < 20 (the product is clearly off-topic).

For "{query}" specifically: prioritize products whose TITLE or AI_TAGS make the intent explicit (the right occasion / style / material / category). Penalize products whose category is wrong (e.g. an accessory when the user wants a dress, a smartwatch when the user wants a classic watch).

Products (JSON):
{items}

Respond with a JSON array. Exactly one object per product, in input order:
[
  {{"idx": 0, "score": 87, "drop": false}},
  ...
]

Output ONLY the JSON array. No markdown, no commentary."""


def _rerank_cache_key(query: str, candidate_ids: list[int]) -> tuple:
    qh = _query_hash(query)
    ch = hashlib.sha256(
        ",".join(str(i) for i in sorted(candidate_ids)).encode()
    ).hexdigest()
    return (qh, ch)


def _rerank_cache_get(key: tuple):
    return _RERANK_CACHE.get(key)


def _rerank_cache_put(key: tuple, val: list[dict]) -> None:
    _RERANK_CACHE[key] = val
    _RERANK_CACHE_ORDER.append(key)
    while len(_RERANK_CACHE_ORDER) > _RERANK_CACHE_MAX:
        old = _RERANK_CACHE_ORDER.pop(0)
        _RERANK_CACHE.pop(old, None)


async def rerank_with_gemini(
    query: str,
    candidates: list[dict],
    *,
    timeout_seconds: float = 5.0,
) -> list[dict]:
    """Re-rank `candidates` (each dict needs at minimum {idx, title,
    ai_tags, product_type}) using a single Gemini call.

    Returns a list of `{idx, score, drop}` dicts. On any failure
    (no API key, Gemini error, timeout, JSON parse) returns the
    identity ranking [{idx: i, score: 50, drop: False} ...] so the
    caller can keep the upstream hybrid order.
    """
    if not candidates:
        return []

    identity = [
        {"idx": i, "score": 50, "drop": False} for i in range(len(candidates))
    ]

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return identity

    candidate_ids = [c.get("id") for c in candidates if c.get("id") is not None]
    if candidate_ids:
        cache_key = _rerank_cache_key(query, candidate_ids)
        cached = _rerank_cache_get(cache_key)
        if cached is not None:
            return cached
    else:
        cache_key = None

    # Trim each candidate dict to just the fields Gemini needs — keeps
    # the prompt small and avoids leaking unrelated DB columns.
    items_for_prompt = [
        {
            "idx": i,
            "title": (c.get("title") or "")[:200],
            "ai_tags": (c.get("ai_tags") or "")[:300],
            "product_type": (c.get("product_type") or "")[:80],
            "subniche": (c.get("subniche") or "")[:30],
        }
        for i, c in enumerate(candidates)
    ]

    prompt = _RERANK_PROMPT_TEMPLATE.format(
        query=query,
        n=len(items_for_prompt),
        items=json.dumps(items_for_prompt, ensure_ascii=False),
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }

    for model in [_MODEL, *_FALLBACK_MODELS]:
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                r = await client.post(
                    _gemini_url(model),
                    params={"key": api_key},
                    json=payload,
                )
            if r.status_code != 200:
                logger.warning(
                    "rerank_with_gemini: %s returned HTTP %d: %s",
                    model, r.status_code, r.text[:200],
                )
                continue
            data = r.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    continue
            if not isinstance(parsed, list):
                continue
            # Validate + repair: ensure every input index is represented
            seen_indices = set()
            out: list[dict] = []
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                idx = entry.get("idx")
                if not isinstance(idx, int) or idx in seen_indices:
                    continue
                if idx < 0 or idx >= len(candidates):
                    continue
                seen_indices.add(idx)
                score = entry.get("score", 50)
                if not isinstance(score, (int, float)):
                    score = 50
                drop = bool(entry.get("drop", False))
                out.append({"idx": idx, "score": float(score), "drop": drop})
            # Fill in any missing indices with neutral scores so the
            # caller never loses a candidate to a Gemini bug.
            for i in range(len(candidates)):
                if i not in seen_indices:
                    out.append({"idx": i, "score": 50.0, "drop": False})
            if cache_key is not None:
                _rerank_cache_put(cache_key, out)
            return out
        except Exception as e:
            logger.warning("rerank_with_gemini: %s exception %s", model, e)
            continue

    return identity


# ---------------------------------------------------------------------
# Scoring helpers — pure functions, no I/O.
# ---------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9\-']{1,}", re.UNICODE)


def _tokenise(text: str) -> set[str]:
    """Lowercased token set for fast set-overlap scoring. Hyphenated
    words (off-shoulder) stay intact. Single chars dropped."""
    if not text:
        return set()
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def score_product_against_expansion(
    *,
    title: str,
    ai_tags: str,
    product_category: str,
    subniche: str,
    product_type: str,
    handle: str,
    exp: ExpansionResult,
) -> float:
    """Score one product against an expansion. Higher = more relevant.

    Weights chosen so that:
      - An exact phrase match in title trumps everything else (+15)
      - Tag overlaps in ai_tags are worth a lot (+4 each, the
        classifier curates these)
      - Title token overlaps are next (+3 each)
      - product_category / subniche / product_type / handle hits
        are weaker bonuses (+1 each)

    Returns 0 if NOTHING overlaps — the caller can drop those.
    """
    score = 0.0
    title_lc = (title or "").lower()
    ai_tags_lc = (ai_tags or "").lower()

    # Exact phrase match in title is the strongest signal — when a user
    # types "wrap dress" and a product is literally "Black Wrap Dress",
    # rank it first.
    original_phrase = exp.original.strip().lower()
    if original_phrase and len(original_phrase) >= 3 and original_phrase in title_lc:
        score += 15

    # Semantic phrases ("floor-length gown", "evening gown")
    for phrase in exp.semantic_phrases:
        if len(phrase) >= 4 and phrase in title_lc:
            score += 6
        elif len(phrase) >= 4 and phrase in ai_tags_lc:
            score += 4

    title_tokens = _tokenise(title)
    ai_tag_tokens = _tokenise(ai_tags)
    category_tokens = _tokenise(product_category)
    subniche_tokens = _tokenise(subniche)
    product_type_tokens = _tokenise(product_type)
    handle_tokens = _tokenise(handle)

    # Controlled-vocabulary tags — match these in ai_tags FIRST (the
    # classifier emits canonical tokens there) then title.
    all_tags = exp.tag_terms()
    for tag in all_tags:
        if tag in ai_tag_tokens:
            score += 4
        if tag in title_tokens:
            score += 2

    # Canonical English synonyms and multilingual nouns — match
    # generously against everything searchable.
    all_terms = (
        set(t.lower() for t in exp.canonical_terms)
        | set(t.lower() for t in exp.multilingual_nouns)
    )
    # Original query tokens
    for t in re.split(r"\s+", exp.original.lower()):
        if len(t) >= 3:
            all_terms.add(t)

    for term in all_terms:
        # Multi-word terms ("evening gown"): substring in title
        if " " in term or "-" in term:
            if len(term) >= 4:
                if term in title_lc:
                    score += 3
                elif term in ai_tags_lc:
                    score += 2
            continue
        if len(term) < 3:
            continue
        if term in title_tokens:
            score += 3
        if term in ai_tag_tokens:
            score += 2
        if term in category_tokens:
            score += 1
        if term in subniche_tokens:
            score += 1
        if term in product_type_tokens:
            score += 1
        if term in handle_tokens:
            score += 1

    return score


# ---------------------------------------------------------------------
# Diagnostics — used by /api/debug/search to surface what the
# expander produced and how each candidate scored.
# ---------------------------------------------------------------------
def expansion_to_dict(exp: ExpansionResult) -> dict:
    return {
        "original": exp.original,
        "canonical_terms": exp.canonical_terms,
        "occasion_tags": exp.occasion_tags,
        "style_tags": exp.style_tags,
        "material_tags": exp.material_tags,
        "color_tags": exp.color_tags,
        "multilingual_nouns": exp.multilingual_nouns,
        "semantic_phrases": exp.semantic_phrases,
        "cached": exp.cached,
        "expander_used": exp.expander_used,
    }
