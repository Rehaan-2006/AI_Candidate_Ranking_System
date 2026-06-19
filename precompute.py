# precompute.py
"""
Pre-computation step — run once, no time limit.
Produces:
  cache/candidate_embeddings.npy   (N × 768, float32)
  cache/jd_embedding.npy           (768,)
  cache/candidate_features.pkl     (candidates list + features list)
"""

import gzip
import json
import os
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from features import extract_features

CANDIDATES_FILE = "candidates.jsonl.gz"
CACHE_DIR       = "cache"
MODEL_NAME      = "BAAI/bge-base-en-v1.5"
BATCH_SIZE      = 256

# ── JD Query ─────────────────────────────────────────────────────────────────
# Distilled from JD "must-haves" + "ideal candidate" paragraph.
# Intentionally includes both explicit keywords AND semantic descriptions
# so plain-language Tier 5 candidates surface correctly.
JD_QUERY = """
Senior AI engineer with 5 to 9 years experience, primarily at product companies not IT services firms.
Production experience building embeddings-based retrieval systems using sentence-transformers, BGE, E5, or similar models.
Hands-on with vector databases and hybrid search: Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch.
Shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.
Strong Python and production ML engineering — not pure research, not just demos.
Designed evaluation frameworks for ranking systems: NDCG, MRR, MAP, A/B testing, offline to online correlation.
Experience with LLM integration: fine-tuning, LoRA, QLoRA, PEFT, knowing when to fine-tune versus prompting.
NLP and information retrieval background. Not primarily computer vision, speech, or robotics.
Located in Pune, Noida, Hyderabad, Mumbai, or Delhi NCR, India. Open to relocation.
Actively looking for work, responsive to recruiters, short notice period preferred.
"""


def build_candidate_text(candidate):
    """
    Builds the richest possible text representation of a candidate for embedding.
    Order: job descriptions (highest signal) → verified skills → summary.
    """
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])
    signals  = candidate.get("redrob_signals", {})
    assessed = signals.get("skill_assessment_scores", {})

    parts = []

    # Context header
    parts.append(
        f"{profile.get('current_title', '')} at {profile.get('current_company', '')}. "
        f"{profile.get('years_of_experience', 0)} years of experience. "
        f"Industry: {profile.get('current_industry', '')}."
    )

    # Career descriptions — the real signal
    for job in career[:4]:
        desc = job.get("description", "").strip()
        if desc:
            parts.append(
                f"At {job.get('company', '')} as {job.get('title', '')} "
                f"({job.get('industry', '')}, {job.get('company_size', '')}): {desc}"
            )

    # Only verified or long-duration advanced skills
    for skill in skills:
        name       = skill.get("name", "")
        prof       = skill.get("proficiency", "")
        dur        = skill.get("duration_months", 0)
        assessed_v = assessed.get(name)

        if prof in ["advanced", "expert"] and dur >= 12:
            if assessed_v and assessed_v >= 60:
                parts.append(
                    f"Verified {name} expertise: assessed {assessed_v:.0f}/100 "
                    f"with {dur} months practical experience."
                )
            elif not assessed_v:
                parts.append(f"Advanced {name}, {dur} months experience.")

    # Summary last (tends to be generic/templated)
    summary = profile.get("summary", "").strip()
    if summary:
        parts.append(summary)

    return " ".join(parts)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # ── Load candidates ──────────────────────────────────────────────────────
    print(f"Loading candidates from {CANDIDATES_FILE} ...")
    candidates = []
    with gzip.open(CANDIDATES_FILE, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing JSONL"):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates")

    # ── Build candidate texts ────────────────────────────────────────────────
    print("Building candidate texts...")
    texts = [build_candidate_text(c) for c in tqdm(candidates, desc="Text builder")]

    # ── Embed candidates ─────────────────────────────────────────────────────
    print(f"Embedding {len(texts):,} candidates (batch_size={BATCH_SIZE})...")
    candidate_embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2 norm → dot product == cosine sim
        convert_to_numpy=True,
    )

    # ── Embed JD ─────────────────────────────────────────────────────────────
    # BGE asymmetric retrieval: prepend instruction to the query side only
    print("Embedding JD query...")
    jd_embedding = model.encode(
        f"Represent this sentence for searching relevant passages: {JD_QUERY}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # ── Extract structured features ──────────────────────────────────────────
    print("Extracting features...")
    features = [
        extract_features(c)
        for c in tqdm(candidates, desc="Features")
    ]

    # ── Save to cache ────────────────────────────────────────────────────────
    emb_path  = f"{CACHE_DIR}/candidate_embeddings.npy"
    jd_path   = f"{CACHE_DIR}/jd_embedding.npy"
    feat_path = f"{CACHE_DIR}/candidate_features.pkl"

    np.save(emb_path, candidate_embeddings)
    np.save(jd_path,  jd_embedding)

    with open(feat_path, "wb") as f:
        pickle.dump({"candidates": candidates, "features": features}, f, protocol=4)

    print("\n✅ Pre-computation complete!")
    print(f"   Embeddings : {candidate_embeddings.shape}  →  {emb_path}")
    print(f"   JD vector  : {jd_embedding.shape}          →  {jd_path}")
    print(f"   Features   : {len(features)} records       →  {feat_path}")


if __name__ == "__main__":
    main()