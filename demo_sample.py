# demo_sample.py
"""
Usage:
    python3 demo_sample.py --input sample_candidates.json --top 10
    python3 demo_sample.py --input candidates.jsonl --limit 500 --top 10
"""

import argparse
import csv
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from features import extract_features
from scorer import compute_scores
from reasoning import generate_reasoning

MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Same JD query used in precompute.py — kept identical so demo results
# are consistent with the real submission, not a separate approximation.
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
    """Identical to the text builder in precompute.py — career history first,
    then verified skills, then summary last."""
    profile  = candidate.get("profile", {})
    career   = candidate.get("career_history", [])
    skills   = candidate.get("skills", [])
    signals  = candidate.get("redrob_signals", {})
    assessed = signals.get("skill_assessment_scores", {})

    parts = [
        f"{profile.get('current_title', '')} at {profile.get('current_company', '')}. "
        f"{profile.get('years_of_experience', 0)} years of experience. "
        f"Industry: {profile.get('current_industry', '')}."
    ]

    for job in career[:4]:
        desc = job.get("description", "").strip()
        if desc:
            parts.append(
                f"At {job.get('company', '')} as {job.get('title', '')} "
                f"({job.get('industry', '')}, {job.get('company_size', '')}): {desc}"
            )

    for skill in skills:
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        dur  = skill.get("duration_months", 0)
        assessed_v = assessed.get(name)
        if prof in ["advanced", "expert"] and dur >= 12:
            if assessed_v and assessed_v >= 60:
                parts.append(
                    f"Verified {name} expertise: assessed {assessed_v:.0f}/100 "
                    f"with {dur} months practical experience."
                )
            elif not assessed_v:
                parts.append(f"Advanced {name}, {dur} months experience.")

    summary = profile.get("summary", "").strip()
    if summary:
        parts.append(summary)

    return " ".join(parts)


def load_candidates(path, limit=None):
    """Handles both pretty-printed JSON arrays (sample_candidates.json)
    and line-delimited JSONL (candidates.jsonl)."""
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
                if limit and len(out) >= limit:
                    break
        return out
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data[:limit] if limit else data


def main():
    parser = argparse.ArgumentParser(description="Sandbox demo: full pipeline on a small sample")
    parser.add_argument("--input", default="sample_candidates.json",
                        help="Path to a JSON array or .jsonl file of candidates")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on how many candidates to load")
    parser.add_argument("--top", type=int, default=10,
                        help="How many top candidates to print/save")
    args = parser.parse_args()

    t0 = time.time()

    print(f"Loading candidates from {args.input} ...")
    candidates = load_candidates(args.input, args.limit)
    print(f"Loaded {len(candidates)} candidates")

    print(f"Loading model: {MODEL_NAME} (CPU)")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("Building candidate texts...")
    texts = [build_candidate_text(c) for c in candidates]

    print("Embedding candidates + job description...")
    candidate_embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    jd_embedding = model.encode(
        f"Represent this sentence for searching relevant passages: {JD_QUERY}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("Extracting features (career quality, availability, location, skill depth, honeypot)...")
    features_list = [extract_features(c) for c in candidates]

    print("Scoring...")
    semantic_scores = candidate_embeddings @ jd_embedding
    final_scores = compute_scores(semantic_scores, features_list)

    candidate_ids = np.array([c["candidate_id"] for c in candidates])
    order = np.lexsort((candidate_ids, -final_scores))
    top_k = min(args.top, len(candidates))
    top_indices = order[:top_k]

    print(f"\n{'='*70}")
    print(f"TOP {top_k} OF {len(candidates)} SAMPLE CANDIDATES")
    print(f"{'='*70}\n")

    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        c = candidates[idx]
        feat = features_list[idx]
        score = float(final_scores[idx])
        reasoning = generate_reasoning(c, rank, score, feat)

        print(f"#{rank}  {c['candidate_id']}  (score: {score:.4f})")
        print(f"    {reasoning}\n")

        rows.append({
            "candidate_id": c["candidate_id"],
            "rank": rank,
            "score": round(score, 6),
            "reasoning": reasoning,
        })

    with open("demo_output.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"{'='*70}")
    print(f"Done in {elapsed:.1f}s on {len(candidates)} candidates → demo_output.csv")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()