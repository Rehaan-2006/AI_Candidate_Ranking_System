# System Architecture Note
## Redrob Intelligent Candidate Ranking Engine

---

### Problem Framing

Given a single job description and a pool of 100,000 candidate profiles, the system must produce a ranked shortlist of the top 100 candidates, with a 1–2 sentence explanation per candidate. The core challenge is going beyond keyword matching — a candidate who "built a recommendation engine at scale" should surface ahead of one who lists "RAG" and "Pinecone" in their skills but has weak assessed scores and an IT-services-only career.

Two hard constraints shape the architecture: the ranking step must complete in under 5 minutes on CPU with no network access, and honeypot profiles (structurally impossible candidates designed to trap naive rankers) must be suppressed — any submission with more than 10 honeypots in the top 100 is disqualified.

---

### Two-Phase Architecture

The system is split into a pre-computation phase (no time limit) and a ranking phase (≤5 minutes). This split is the foundational design decision: it allows a high-quality embedding model to be used without any speed penalty at ranking time.

---

### Phase 1 — Pre-computation

**Input:** `candidates.jsonl` (100,000 candidate profiles)

**Step 1.1 — Text Construction (`precompute.py: build_candidate_text`)**

Each candidate profile is converted into a single rich text string. The ordering is intentional and signal-weighted:

1. Current role context (title, company, years of experience, industry)
2. Career history descriptions — the highest-signal content, where actual work done is described
3. Verified, long-duration skills only (advanced/expert proficiency, ≥12 months, assessed ≥60/100)
4. Profile summary last (tends to be templated and keyword-heavy)

This ordering ensures that a candidate whose career description says "architected a vector search pipeline serving 10M daily queries" produces an embedding that captures semantic fit, regardless of whether they used the word "retrieval" in their skills list.

**Step 1.2 — Candidate Embedding**

Model: `BAAI/bge-base-en-v1.5` (768-dimensional dense vectors, ~109M parameters)

All 100,000 candidate texts are encoded in batches with L2 normalization enabled. The result is a `(100000, 768)` float32 numpy array saved to `cache/candidate_embeddings.npy` (~293 MB).

BGE was selected over lighter alternatives (`all-MiniLM-L6-v2`, `bge-small`) because this is an asymmetric retrieval task — short query against long documents — and BGE-base is specifically trained for this pattern. Since pre-computation has no time constraint, the larger model costs nothing in the ranking step.

**Step 1.3 — JD Embedding**

The job description is distilled into a focused query string (not the full 3-page JD) covering the role's must-haves and ideal candidate traits. BGE's asymmetric retrieval instruction prefix is prepended to the JD embedding only, following the model's intended usage pattern. The result is a `(768,)` vector saved to `cache/jd_embedding.npy`.

**Step 1.4 — Structured Feature Extraction (`features.py`)**

For each candidate, five scalar feature scores (all normalized to [0,1]) are extracted:

- **Career Quality (0–1):** Penalizes candidates whose entire career is at IT services/consulting firms (hard disqualifier, returns 0.05). Rewards product company experience, consistent AI/ML title progression, and tenure stability. Penalizes title-chasing (avg tenure < 12 months) and candidates whose skills are dominated by computer vision or speech — domains the JD explicitly disqualifies.

- **Availability (0–1):** Weighted combination of five platform signals — recency of last login (exponential decay, 90-day half-life), open-to-work flag, recruiter response rate, notice period (≤30 days ideal; >90 days penalized heavily), and interview completion rate.

- **Location (0–1):** Tiered scoring against the role's Pune/Noida preference. Hyderabad/Mumbai/Delhi NCR scores 0.90. Other Indian cities score 0.65 with relocation willingness or 0.50 without. Outside India scores 0.35 (willing to relocate) or 0.10 (not willing).

- **Skill Depth (0–1):** Scores only JD-relevant skills (vector DBs, embeddings, retrieval, LLM integration, NLP, evaluation). For each relevant skill, the contribution is weighted by verified assessment score and duration in months. Claimed "advanced/expert" proficiency with an assessment score below 50 triggers a keyword-stuffer penalty. GitHub activity score (normalized) contributes 25% of the final skill depth score.

- **Honeypot Multiplier (0.05–1.0):** Applied after all scoring is complete. Flags structural impossibilities: expert skills with zero months of duration, implausibly many expert skills (>8), systematic proficiency inflation versus verified assessments, career duration inconsistencies, education timeline paradoxes (master's degree starting before bachelor's finishes), and non-technical current titles paired with rich AI skill lists. Each flag increments a counter; the counter maps to a multiplier from 1.0 (clean) down to 0.05 (near-zero, effectively eliminated).

All feature dicts for all 100,000 candidates are serialized to `cache/candidate_features.pkl`.

---

### Phase 2 — Ranking (≤5 minutes)

**Input:** Three cache files from Phase 1, plus the JD embedding.

**Step 2.1 — Load Artifacts (~2–4 seconds)**

`candidate_embeddings.npy`, `jd_embedding.npy`, and `candidate_features.pkl` are loaded into memory. Total RAM footprint: ~1.5 GB.

**Step 2.2 — Semantic Similarity (~1 second)**

Because both embedding arrays are L2-normalized, dot product equals cosine similarity. A single matrix multiplication computes all 100,000 similarity scores simultaneously:

```
semantic_scores = candidate_embeddings @ jd_embedding   # shape: (100000,)
```

No vector database, no FAISS index, no approximate nearest-neighbor search — just numpy. For this workload (single fixed query, static corpus), it is the fastest and simplest possible approach.

**Step 2.3 — Weighted Score Fusion (`scorer.py`)**

The final composite score for each candidate:

```
score = (0.35 × semantic)
      + (0.25 × career_quality)
      + (0.20 × availability)
      + (0.15 × location)
      + (0.05 × skill_depth)
      × honeypot_multiplier
```

Weights are calibrated for the challenge's evaluation metric (NDCG@10 is 50% of the total score), so getting the top 10 right matters more than uniformly distributing quality across all 100.

**Step 2.4 — Top-100 Selection with Tie-Breaking**

`np.lexsort` sorts candidates by score descending, with `candidate_id` ascending as a stable tie-breaker:

```python
order = np.lexsort((candidate_ids, -final_scores))
top_indices = order[:100]
```

This satisfies the submission validator's requirement that equal-scored candidates appear in ascending `candidate_id` order.

**Step 2.5 — Reasoning Generation (`reasoning.py`)**

For each of the top 100 candidates, a 1–2 sentence reasoning string is assembled from actual profile and signal fields — no hallucination, no LLM call at ranking time. Sentence 1 surfaces the top strengths (years of experience, best product company, top verified skill, career trajectory). Sentence 2 surfaces notable concerns (notice period above 30 days, inactivity, location mismatch, low response rate) or highlights availability positives if there are no concerns. Every claim in the reasoning maps to a specific field in the candidate record.

**Step 2.6 — CSV Output**

Columns: `candidate_id`, `rank` (1–100), `score` (monotonically non-increasing), `reasoning`. Validated by `validate_submission.py` before submission.

---

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Embedding model | `BAAI/bge-base-en-v1.5` via `sentence-transformers` |
| Embedding storage | NumPy `.npy` file (~293 MB) |
| Feature storage | Python `pickle` (`.pkl`) |
| Semantic similarity | NumPy matrix multiply |
| Feature extraction | Pure Python |
| Output format | Python `csv` stdlib |
| GPU pre-computation | Google Colab (T4), downloadable cache for local ranking |

---

### What Makes This Non-Trivially Hard

**The honeypot problem:** A pure BM25 or TF-IDF ranker would score keyword stuffers highly. A pure semantic ranker would surface honeypots with plausible career descriptions. The honeypot detector specifically targets structural data inconsistencies that no amount of text similarity can catch — impossible education timelines, claimed expert proficiency with zero duration in a skill, and assessment scores that dramatically contradict stated proficiency.

**The plain-language Tier 5 problem:** Genuinely excellent candidates who don't use buzzwords must surface. Embedding career descriptions (not just skill lists) into the query space addresses this directly — "scaled a search infrastructure from 100K to 10M daily queries" and "built Milvus-backed semantic search" embed into similar regions of the 768-dimensional space.

**The availability-vs-fit tradeoff:** A semantically perfect candidate who last logged in 9 months ago, won't relocate from a non-preferred city, and has a 120-day notice period is, practically speaking, close to unhireable. The availability and location scores weight these behavioral signals meaningfully without completely overriding strong semantic and career quality signals.