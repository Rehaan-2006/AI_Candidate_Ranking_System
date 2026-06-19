# features.py
"""
Extracts structured feature scores for each candidate.
All component scores are normalized to [0, 1].
"""

import math
from datetime import datetime, date

from honeypot import detect_honeypot

# ── Constants ───────────────────────────────────────────────────────────────

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "wipro", "infosys", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
    "mphasis", "hexaware", "ltimindtree", "l&t infotech", "birlasoft",
}

PREFERRED_LOCATIONS = {
    "pune": 1.00, "noida": 1.00,
    "hyderabad": 0.90, "mumbai": 0.90, "delhi": 0.90,
    "gurugram": 0.90, "gurgaon": 0.90, "new delhi": 0.90,
    "bangalore": 0.75, "bengaluru": 0.75,
    "chennai": 0.70, "kolkata": 0.65,
}

# Skills directly referenced in JD must-haves or ideal candidate section
JD_SKILLS = {
    # Vector DBs / retrieval infra
    "faiss", "milvus", "pinecone", "weaviate", "qdrant",
    "opensearch", "elasticsearch", "pgvector",
    # Retrieval / embeddings
    "embeddings", "sentence-transformers", "dense retrieval",
    "hybrid search", "information retrieval", "bm25",
    # LLM / ranking
    "llms", "fine-tuning llms", "lora", "qlora", "peft", "rag",
    "ranking", "recommendation systems", "learning to rank",
    "haystack", "langchain",
    # Core ML / NLP
    "nlp", "transformers", "hugging face", "hugging face transformers",
    "pytorch", "python", "scikit-learn",
    # Evaluation
    "mlops", "a/b testing", "ndcg", "evaluation",
}

# Domains the JD explicitly disqualifies as primary expertise
CV_SKILLS = {
    "computer vision", "image classification", "object detection",
    "yolo", "resnet", "cnn", "image segmentation", "ocr",
    "tts", "speech recognition", "gans", "diffusion models",
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _days_since(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return 999


def _is_consulting(company_name):
    name = company_name.lower()
    return any(firm in name for firm in CONSULTING_FIRMS)


# ── Component scorers ────────────────────────────────────────────────────────

def career_quality(candidate):
    """
    Scores career trajectory and company quality.
    Hard disqualifiers applied first, then additive bonuses.
    """
    career  = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    skills  = candidate.get("skills", [])

    if not career:
        return 0.10

    # Hard disqualifier: entire career at consulting firms
    if all(_is_consulting(j.get("company", "")) for j in career):
        return 0.05

    score = 0.40  # base

    # Product company ratio
    product_jobs = [j for j in career if not _is_consulting(j.get("company", ""))]
    score += 0.20 * (len(product_jobs) / len(career))

    # AI/ML title progression through career
    ai_keywords = [
        "ml", "ai", "machine learning", "nlp", "data scientist",
        "search", "ranking", "retrieval", "recommendation", "research scientist",
    ]
    ai_jobs = [
        j for j in career
        if any(kw in j.get("title", "").lower() for kw in ai_keywords)
    ]
    score += 0.15 * (len(ai_jobs) / len(career))

    # YoE sweet spot: 4–10 years
    yoe = float(profile.get("years_of_experience", 0))
    if 4 <= yoe <= 10:
        score += 0.10
    elif 3 <= yoe < 4 or 10 < yoe <= 12:
        score += 0.05

    # Tenure stability — JD dislikes title-chasers (avg tenure < 18 months)
    tenures = [j.get("duration_months", 0) for j in career]
    avg_tenure = sum(tenures) / len(tenures) if tenures else 0
    if avg_tenure >= 24:
        score += 0.10
    elif avg_tenure >= 18:
        score += 0.05
    elif avg_tenure < 12:
        score -= 0.15

    # CV-primary domain penalty (JD explicit disqualifier)
    all_names = [s.get("name", "").lower() for s in skills]
    cv_count  = sum(1 for n in all_names if n in CV_SKILLS)
    if all_names and (cv_count / len(all_names)) > 0.40:
        score -= 0.12

    # Wrong current domain: marketing, sales, HR, mechanical, etc.
    wrong_domains = [
        "marketing manager", "sales manager", "hr manager",
        "operations manager", "brand manager", "mechanical engineer",
    ]
    current = profile.get("current_title", "").lower()
    if any(d in current for d in wrong_domains):
        score -= 0.30

    return max(0.0, min(1.0, score))


def availability(candidate):
    """
    Scores actual hiring availability from platform signals.
    """
    signals = candidate.get("redrob_signals", {})

    # Recency: exponential decay, 90-day half-life
    days_idle = _days_since(signals.get("last_active_date", "2020-01-01"))
    recency   = math.exp(-days_idle * math.log(2) / 90)

    # Open to work flag
    openness = 1.0 if signals.get("open_to_work_flag", False) else 0.55

    # Recruiter responsiveness
    response = signals.get("recruiter_response_rate", 0.0)

    # Notice period — JD wants ≤30, can buy out 30
    notice_days = signals.get("notice_period_days", 90)
    if notice_days <= 30:
        notice = 1.00
    elif notice_days <= 60:
        notice = 0.75
    elif notice_days <= 90:
        notice = 0.50
    else:
        notice = 0.20   # 120+ days is a significant concern

    # Interview reliability
    interview = signals.get("interview_completion_rate", 0.5)

    return (
        recency   * 0.30 +
        openness  * 0.20 +
        response  * 0.25 +
        notice    * 0.15 +
        interview * 0.10
    )


def location(candidate):
    """
    Scores location fit for a Pune/Noida-preferred role.
    """
    profile    = candidate.get("profile", {})
    signals    = candidate.get("redrob_signals", {})
    loc        = profile.get("location", "").lower()
    country    = profile.get("country", "").lower()
    can_relocate = signals.get("willing_to_relocate", False)

    for city, city_score in PREFERRED_LOCATIONS.items():
        if city in loc:
            return city_score

    if country == "india" or "india" in loc:
        return 0.65 if can_relocate else 0.50

    return 0.35 if can_relocate else 0.10


def skill_depth(candidate):
    """
    Scores depth of JD-relevant skills, weighted by assessment verification.
    Penalises keyword stuffers (claimed advanced but assessed <50).
    """
    skills   = candidate.get("skills", [])
    signals  = candidate.get("redrob_signals", {})
    assessed = signals.get("skill_assessment_scores", {})

    github_raw   = signals.get("github_activity_score", -1)
    github_score = github_raw / 100 if github_raw >= 0 else 0.15

    if not skills:
        return github_score * 0.25

    total, count = 0.0, 0

    for skill in skills:
        name = skill.get("name", "").lower()
        if not any(kw in name for kw in JD_SKILLS):
            continue

        count += 1
        prof     = skill.get("proficiency", "beginner")
        dur      = skill.get("duration_months", 0)
        prof_map = {"beginner": 0.20, "intermediate": 0.50, "advanced": 0.80, "expert": 1.00}
        base     = prof_map.get(prof, 0.20)
        dur_score = min(dur / 36, 1.0)

        assessed_val = assessed.get(skill.get("name", ""))
        if assessed_val is not None:
            verified = assessed_val / 100
            # Keyword stuffer penalty
            if prof in ["advanced", "expert"] and assessed_val < 50:
                verified *= 0.50
            contribution = verified * 0.70 + dur_score * 0.30
        else:
            contribution = base * 0.60 + dur_score * 0.40

        total += contribution

    if count == 0:
        return github_score * 0.25

    avg = total / count
    return avg * 0.75 + github_score * 0.25


# ── Main entry point ─────────────────────────────────────────────────────────

def extract_features(candidate):
    """Extract all features. Returns a dict."""
    return {
        "candidate_id":      candidate["candidate_id"],
        "career_quality":    career_quality(candidate),
        "availability":      availability(candidate),
        "location":          location(candidate),
        "skill_depth":       skill_depth(candidate),
        "honeypot_multiplier": detect_honeypot(candidate),
    }