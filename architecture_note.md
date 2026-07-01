# System Architecture Note
## Redrob Intelligent Candidate Discovery & Ranking Engine

---

## Problem framing

The challenge: rank 100,000 candidate profiles against a single job description, within a 5-minute CPU constraint, with explainable per-candidate output.

Three forces make this non-trivial:
1. **Scale** — embedding 100k candidates at inference time on CPU is infeasible (~40 min).
2. **Keyword traps** — the dataset deliberately contains profiles that match on skills keywords but are disqualified by career trajectory, domain, or behavioral signals.
3. **Availability gap** — a technically perfect candidate who is unreachable is, for practical hiring, worthless.

---

## Two-phase architecture

The system splits into two decoupled phases with a cache layer between them.

### Phase 1 — Precomputation (offline, no time limit)

**Inputs:** `candidates.jsonl` (100,000 candidates, ~465 MB uncompressed)

**Step 1.1 — Text representation**
Each candidate is converted to a single rich-text string that prioritises the highest-signal fields in this order:
- Header line: current title, company, years of experience, industry
- Job descriptions from career history (first 4 roles), including company, title, industry, and company size context — this is where real semantic signal lives
- Advanced/expert skills with duration ≥ 12 months that have platform-verified assessment scores ≥ 60 — only verified high-confidence skills
- Profile summary (lowest priority — tends to be generic)

Skills with zero duration, unverified high-proficiency claims, and plain-language career narratives are all handled: BGE's training distribution covers paraphrase, so "built a search system at a startup" and "production retrieval engineering" are close in embedding space.

**Step 1.2 — Embedding**
Model: `BAAI/bge-base-en-v1.5` (768-dim, 109M parameters)

Chosen for:
- Asymmetric retrieval design — separate query/passage representations, suited for JD-to-candidate matching
- State-of-the-art performance on BEIR and MTEB retrieval benchmarks
- Practical size — fits on free-tier Colab T4, runs in ~12 min for 100k candidates at batch size 512

Both candidate embeddings and the JD query embedding are L2-normalised at encode time, so cosine similarity reduces to a dot product — enabling a single NumPy matrix multiplication at ranking time.

JD query uses the BGE asymmetric prefix: `"Represent this sentence for searching relevant passages: {JD_QUERY}"`. Candidate texts do not use the prefix (passage side).

**Step 1.3 — Feature extraction**
Five structured scores computed per candidate, each normalised to [0, 1]:

| Feature | Key signals used |
|---|---|
| `career_quality` | Product vs consulting ratio, AI/ML title progression, YoE sweet spot (4–10yr), avg tenure stability, CV-primary domain penalty, wrong-domain current title penalty |
| `availability` | Platform recency (90-day exponential half-life), open-to-work flag, recruiter response rate, notice period bracket, interview completion rate |
| `location` | City-level scoring (Pune/Noida = 1.0, Hyderabad/Mumbai/Delhi = 0.9, Bangalore = 0.75, etc.), relocation willingness fallback |
| `skill_depth` | Per-skill contribution: proficiency level × duration × assessment verification; keyword-stuffer penalty (claimed advanced/expert but assessed < 50); GitHub activity as supplemental signal |
| `honeypot_multiplier` | See Honeypot Detection section below |

**Step 1.4 — Cache write**
Three files written to `cache/`:
- `candidate_embeddings.npy` — shape (100000, 768), float32, L2-normalised
- `jd_embedding.npy` — shape (768,), L2-normalised
- `candidate_features.pkl` — dict with `"candidates"` (raw dicts) and `"features"` (scored dicts), aligned by index

---

### Phase 2 — Ranking (online, ≤5 min CPU constraint)

**Runtime:** ~4 seconds on CPU for 100k candidates.

**Step 2.1 — Semantic scores**
```python
semantic_scores = candidate_embeddings @ jd_embedding  # shape (100000,)
```
Single matrix multiplication. Because both sides are L2-normalised, this equals cosine similarity. No loop, no GPU needed.

**Step 2.2 — Weighted composite**
```python
final_score = (
    0.35 * semantic_score +
    0.25 * career_quality +
    0.20 * availability +
    0.15 * location +
    0.05 * skill_depth
) * honeypot_multiplier
```

Weight rationale:
- Semantic (35%): primary fit signal, captures both explicit and paraphrased relevance
- Career quality (25%): most durable signal — career trajectory doesn't lie the way skill lists do
- Availability (20%): directly affects whether a hire actually closes
- Location (15%): hard operational constraint for this role
- Skill depth (5%): supplemental verification signal; deliberately low-weighted to avoid keyword-stuffing exploitation

**Step 2.3 — Selection and tie-breaking**
```python
order = np.lexsort((candidate_ids, -final_scores))
top_indices = order[:100]
```
Primary sort: score descending. Tie-break: `candidate_id` ascending (CAND_XXXXXXX lexicographic order). This satisfies the validator's monotonicity and tie-break requirements deterministically.

**Step 2.4 — Reasoning generation**
For each of the top 100, a 1–2 sentence reasoning string is generated directly from the candidate's raw profile fields. Every claim is traced to a field value — no language model, no inference, no hallucination risk. Structure: Sentence 1 = top strengths (YoE, best product company, best verified skill, career progression); Sentence 2 = concerns or availability highlights.

**Step 2.5 — Output**
`submission.csv` with columns: `candidate_id, rank, score, reasoning`. Score is monotonically non-increasing by rank. All 100 ranks covered exactly once.

---

## Honeypot detection

The dataset contains ~80 honeypot profiles with deliberately impossible or fraudulent signals. Submissions with >10% honeypots in top 100 are disqualified.

Detection uses 7 independent flag types, each adding flag points:

| Flag | Signal | Weight |
|---|---|---|
| Zero-duration advanced/expert skill | Claimed expertise with no time spent | +2 |
| Excessive expert count (>8) | Implausible breadth of mastery | +2 |
| Systematic proficiency inflation | Advanced/expert claimed but platform assessment < 40 | +1 each (cap 3) |
| Career duration impossibility | Stated YoE >> sum of all job durations | +2 |
| Single role >130 months | Synthetic data artifact | +1 |
| Education timeline impossibility | Master's started before Bachelor's ended | +2 |
| Domain mismatch | Non-technical current title + rich AI skill list | +3 |

Flag score → multiplier:
```
0 flags → 1.00    (clean)
1 flag  → 0.85
2 flags → 0.65
3 flags → 0.40
4 flags → 0.20
5+      → 0.05    (effectively suppressed)
```

Multiplier is applied to the composite score rather than hard-removing candidates. This avoids catastrophic ranking failures from false positives — a clean candidate with one unusual-but-real signal gets a 15% penalty, not disqualification.

The consulting-firm list used in `career_quality` and `reasoning` is defined once in `constants.py` and imported by both modules, preventing the list from drifting between scoring and explanation.

---

## What the system handles correctly

**Keyword trap candidates:** A candidate with FAISS, Milvus, and RAG listed as expert skills but whose career is 100% at TCS/Wipro receives `career_quality = 0.05` (all-consulting disqualifier), reducing their composite score to roughly 0.03–0.04 regardless of semantic similarity. They do not appear in the top 100.

**Plain-language Tier 5 candidates:** A candidate whose job description reads "built the internal search system for product discovery" without using "dense retrieval" or "vector database" explicitly still embeds close to the JD query because BGE's training covers semantic paraphrase. They surface on semantic score alone.

**Unavailable candidates:** A candidate with an ideal skill profile who hasn't been active for 180 days, has a 5% recruiter response rate, and a 120-day notice period receives `availability ≈ 0.15`. With 20% weight, this knocks ~17 percentage points off their composite score — enough to push them out of the top 100 in a competitive pool.

**Behavioral twins:** Two candidates with identical skill profiles are separated by availability, location, and behavioral signals, ensuring the ranking is not arbitrary at tie points.

---

## Files and dependencies

| File | Role |
|---|---|
| `constants.py` | Single source of truth for consulting-firm list and `is_consulting()` |
| `honeypot.py` | Flag-based multiplier computation |
| `features.py` | Five structured feature scores |
| `scorer.py` | Weighted composite formula |
| `reasoning.py` | Fact-grounded reasoning string generation |
| `precompute.py` | Phase 1 driver |
| `rank.py` | Phase 2 driver |
| `demo_sample.py` | Demo/sandbox — runs on sample_candidates.json, no cache required |
| `redrob_precompute.ipynb` | Colab notebook for cloud GPU precomputation |

**Runtime dependencies:** `sentence-transformers`, `numpy`, `tqdm`  
**Precompute hardware:** GPU recommended (T4 free Colab: ~12 min for 100k candidates)  
**Ranking hardware:** CPU only (no GPU, no network, 16 GB RAM constraint satisfied)