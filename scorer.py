# scorer.py
"""
Computes the final weighted composite score for each candidate.
"""

import numpy as np

# Weights — must sum to 1.0
# Tuned for NDCG@10 (50% of final score) — top 10 quality is paramount
WEIGHTS = {
    "semantic":       0.35,
    "career_quality": 0.25,
    "availability":   0.20,
    "location":       0.15,
    "skill_depth":    0.05,
}


def compute_scores(semantic_scores: np.ndarray, features_list: list) -> np.ndarray:
    """
    Args:
        semantic_scores:  np.array (N,) — cosine similarity of candidate vs JD
        features_list:    list of dicts from extract_features()

    Returns:
        np.array (N,) of final weighted scores, honeypot multiplier applied.
    """
    n = len(features_list)
    final = np.zeros(n, dtype=np.float32)

    for i, feat in enumerate(features_list):
        raw = (
            WEIGHTS["semantic"]       * float(semantic_scores[i]) +
            WEIGHTS["career_quality"] * feat["career_quality"]    +
            WEIGHTS["availability"]   * feat["availability"]      +
            WEIGHTS["location"]       * feat["location"]          +
            WEIGHTS["skill_depth"]    * feat["skill_depth"]
        )
        final[i] = raw * feat["honeypot_multiplier"]

    return final