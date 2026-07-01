# Redrob Intelligent Candidate Discovery & Ranking Engine

**Redrob AI & Datathon Arena — Track 1 Submission**

A two-phase candidate ranking system that scores all 100,000 candidates against a job description and returns the top 100 with per-candidate reasoning. Runs the ranking stage in under 5 seconds on CPU.

---

## How It Works — Pipeline Overview

The system runs in two stages:

**Stage 1 — Pre-computation** (run once, no time limit):
- Loads all 100,000 candidate profiles
- Builds a rich text representation of each candidate (career history, verified skills, summary)
- Embeds all candidates using `BAAI/bge-base-en-v1.5` (768-dimensional dense vectors)
- Extracts structured feature scores (career quality, availability, location, skill depth, honeypot risk)
- Saves everything to `cache/` as numpy arrays and a pickle file

**Stage 2 — Ranking** (< 5 minutes, CPU only, no network):
- Loads pre-computed embeddings and features from `cache/`
- Computes cosine similarity between the JD embedding and all 100k candidate vectors (single matrix multiply, ~1 second)
- Fuses semantic score with structured feature scores using tuned weights
- Applies honeypot detection multiplier to suppress fraudulent or impossible profiles
- Sorts, selects top 100 with tie-breaking, generates fact-grounded reasoning per candidate
- Outputs a validated `submission.csv`

---

## Architecture overview

```
Phase 1 — Precompute (run once, GPU recommended, no time limit)
  candidates.jsonl
       │
       ├── SentenceTransformer (BAAI/bge-base-en-v1.5)
       │        └── candidate_embeddings.npy  (100k × 768)
       │        └── jd_embedding.npy          (768,)
       │
       └── Feature extraction pipeline
                └── candidate_features.pkl    (candidates + 5 feature scores each)

Phase 2 — Ranking (CPU only, ≤5 min constraint)
  cache/ artifacts
       │
       ├── Semantic score: embeddings @ jd_embedding  (single NumPy matmul)
       ├── Weighted composite: semantic + career_quality + availability + location + skill_depth
       ├── Honeypot multiplier applied (suppress fraudulent/impossible profiles)
       └── Top 100 by score → submission.csv
```

See `architecture_note.md` for full detail.

---

## Repository structure

```
.
├── constants.py          # Shared consulting-firm list + is_consulting(); single source of truth
├── honeypot.py           # Fraud/trap profile detection → multiplier [0.05, 1.0]
├── features.py           # Five structured feature scores per candidate
├── scorer.py             # Weighted composite formula
├── reasoning.py          # Fact-grounded 1-2 sentence reasoning (no hallucination)
├── precompute.py         # Phase 1: embed + extract features, save to cache/
├── rank.py               # Phase 2: load cache, score, output top-100 CSV
├── demo_sample.py        # Demo/sandbox: runs full pipeline on sample_candidates.json
├── redrob_precompute.ipynb  # Colab notebook for GPU precomputation
├── validate_submission.py   # Provided format validator
├── requirements.txt
├── architecture_note.md
├── submission_metadata.yaml
├── job_description.md       # Role being ranked against
├── candidate_schema.json    # Provided: candidate data schema reference
└── sample_candidates.json   # Provided: first 50 candidates for demo/inspection
```

---

## Quickstart

### 1. Install dependencies
```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
```

### 2. Phase 1 — Precompute (run once)
Requires GPU for reasonable speed. Two options:

**Option A — Local (if you have GPU):**
```bash
python3 precompute.py
# Reads candidates.jsonl, writes cache/ directory
```

**Option B — Google Colab (recommended for most users):**
Open `redrob_precompute.ipynb` in Colab, set runtime to T4 GPU, run all cells.
Upload `candidates.jsonl` when prompted. Download the 3 cache files into `cache/`.

### 3. Phase 2 — Rank (CPU, <10 seconds)
```bash
python3 rank.py --candidates candidates.jsonl --out submission.csv
```

### 4. Validate
```bash
python3 validate_submission.py submission.csv
```

### 5. Demo (sandbox / video demo)
```bash
python3 demo_sample.py --input sample_candidates.json --top 10
```
Runs the full ranking pipeline on the 50-candidate sample. No precomputed cache needed — embeds on the fly. Outputs ranked results to terminal and `demo_output.csv`.

---

## Design decisions

### Why two phases?
The 5-minute CPU constraint makes embedding 100k candidates during ranking infeasible (≈40 min on CPU). Precomputing once and caching the L2-normalised embeddings reduces the ranking step to a single matrix multiplication (~0.2s).

### Why BGE (BAAI/bge-base-en-v1.5)?
Best-in-class asymmetric retrieval model at the time of development. The query-side instruction prefix (`Represent this sentence for searching relevant passages: ...`) improves recall for plain-language candidates who don't use technical keywords but have genuinely relevant experience ("built a recommendation system" surfaces correctly against a JD that says "ranking system").

### Why not pure keyword matching?
The JD explicitly calls this out as a trap. A candidate with "FAISS", "Milvus", and "LLMs" in their skills section but whose entire career is at an IT services firm is not the target profile. A candidate whose job descriptions show they shipped a search system at a product startup, even without exact keyword matches, is.

### Feature weights
| Component | Weight | Rationale |
|---|---|---|
| Semantic similarity | 35% | Primary fit signal — covers both explicit and paraphrased relevance |
| Career quality | 25% | Product vs consulting, AI title progression, tenure stability |
| Availability | 20% | Platform recency, open-to-work, notice period, recruiter responsiveness |
| Location | 15% | Pune/Noida preferred; Hyderabad/Mumbai/Delhi NCR acceptable |
| Skill depth | 5% | Verified scores from platform assessments; supplemental signal only |

### Honeypot detection
Eight independent flag types (proficiency/duration contradiction, expert count inflation, assessment score vs claimed level mismatch, career duration impossibility, >130-month single role, education timeline impossibility, non-technical title + rich AI skill list). Flags accumulate into a multiplier: 1.0 → 0.85 → 0.65 → 0.40 → 0.20 → 0.05. Candidates are suppressed, not hard-removed — this keeps the CSV monotonically scored and avoids accidental false-positive disqualification.

---

## Requirements

See `requirements.txt`. Core dependencies:
- `sentence-transformers` — embedding model
- `numpy` — matrix operations
- `tqdm` — progress bars

Python 3.9+ recommended.

---

## AI tools declaration

This submission was developed with AI assistance (Claude by Anthropic) for code generation and review. All architectural decisions, feature weight choices, JD interpretation, and design trade-offs were made by the participant. 