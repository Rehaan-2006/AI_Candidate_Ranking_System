# rank.py
"""
Main ranking script — must complete in ≤5 minutes on CPU.
Loads precomputed artifacts, scores all 100k candidates, outputs top-100 CSV.

Usage:
    python rank.py --candidates candidates.jsonl.gz --out submission.csv
"""

import argparse
import csv
import pickle
import time

import numpy as np

from reasoning import generate_reasoning
from scorer import compute_scores

CACHE_DIR = "cache"
TOP_K     = 100


def main():
    parser = argparse.ArgumentParser(description="Redrob candidate ranker")
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidate pool (used only for validation)")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    t0 = time.time()

    # ── Load precomputed artifacts ───────────────────────────────────────────
    print("Loading precomputed embeddings...")
    candidate_embeddings = np.load(f"{CACHE_DIR}/candidate_embeddings.npy")
    jd_embedding         = np.load(f"{CACHE_DIR}/jd_embedding.npy")

    print("Loading precomputed features...")
    with open(f"{CACHE_DIR}/candidate_features.pkl", "rb") as f:
        cache = pickle.load(f)

    candidates    = cache["candidates"]
    features_list = cache["features"]

    print(f"Loaded {len(candidates):,} candidates  [{time.time()-t0:.1f}s]")

    # ── Semantic similarity ──────────────────────────────────────────────────
    # Both vectors are L2-normalised → dot product == cosine similarity
    print("Computing semantic scores...")
    semantic_scores = candidate_embeddings @ jd_embedding  # shape (N,)

    # ── Weighted composite score ─────────────────────────────────────────────
    print("Computing final scores...")
    final_scores = compute_scores(semantic_scores, features_list)

    # ── Top-100 selection ────────────────────────────────────────────────────
    candidate_ids = np.array([c["candidate_id"] for c in candidates])
    order = np.lexsort((candidate_ids, -final_scores))
    top_indices = order[:TOP_K]

    # ── Build CSV rows ───────────────────────────────────────────────────────
    print("Generating reasoning and writing CSV...")
    rows = []
    for rank, idx in enumerate(top_indices, start=1):
        candidate = candidates[idx]
        feat      = features_list[idx]
        score     = float(final_scores[idx])
        reasoning = generate_reasoning(candidate, rank, score, feat)

        rows.append({
            "candidate_id": candidate["candidate_id"],
            "rank":         rank,
            "score":        round(score, 6),
            "reasoning":    reasoning,
        })

    # ── Validate score monotonicity (spec requirement) ───────────────────────
    for i in range(1, len(rows)):
        assert rows[i]["score"] <= rows[i-1]["score"], \
            f"Score not monotonically non-increasing at rank {i+1}"

    # ── Write CSV ────────────────────────────────────────────────────────────
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.1f}s  →  {args.out}")
    print(f"   Rank 1   : {rows[0]['candidate_id']}  (score={rows[0]['score']:.4f})")
    print(f"   Rank 10  : {rows[9]['candidate_id']}  (score={rows[9]['score']:.4f})")
    print(f"   Rank 100 : {rows[99]['candidate_id']} (score={rows[99]['score']:.4f})")
    print(f"\n   Run validate_submission.py next to check format compliance.")


if __name__ == "__main__":
    main()