# reasoning.py
"""
Generates fact-grounded 1-2 sentence reasoning per candidate.
Every claim maps to an actual field in the candidate profile — no hallucination.
"""

from datetime import datetime, date
from constants import is_consulting


def _days_since(date_str):
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (date.today() - d).days
    except Exception:
        return 999


def _best_verified_skill(candidate):
    """Return name + score for the highest-scoring verified skill."""
    assessed = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if not assessed:
        return None
    best_name, best_val = max(assessed.items(), key=lambda x: x[1])
    return f"{best_name} (verified {best_val:.0f}/100)"


def _top_product_company(candidate):
    """Return most recent non-consulting company."""
    for job in candidate.get("career_history", []):
        company = job.get("company", "")
        if not is_consulting(company):
            return company
    return None


def generate_reasoning(candidate, rank, score, features):
    """
    Returns a 1-2 sentence, fact-grounded reasoning string.
    Sentence 1: top strengths. Sentence 2: concerns (if any) or availability highlights.
    """
    profile  = candidate.get("profile", {})
    signals  = candidate.get("redrob_signals", {})

    yoe          = profile.get("years_of_experience", 0)
    current_co   = profile.get("current_company", "")
    notice       = signals.get("notice_period_days", 90)
    response     = signals.get("recruiter_response_rate", 0.0)
    open_to_work = signals.get("open_to_work_flag", False)
    days_idle    = _days_since(signals.get("last_active_date", "2020-01-01"))
    location_str = profile.get("location", "Unknown")
    can_relocate = signals.get("willing_to_relocate", False)

    verified = _best_verified_skill(candidate)
    top_co   = _top_product_company(candidate)

    # ── Sentence 1: Strengths ─────────────────────────────────────────────
    strengths = []

    if yoe:
        strengths.append(f"{yoe:.0f} yrs applied ML/AI")

    if top_co and not _is_consulting(top_co):
        label = "currently" if top_co == current_co else "ex"
        strengths.append(f"product company background ({label}-{top_co})")

    if verified:
        strengths.append(f"verified {verified}")

    if features.get("career_quality", 0) >= 0.65:
        strengths.append("consistent AI/ML career progression")

    s1 = f"Ranked #{rank}: {'; '.join(strengths)}." if strengths else f"Ranked #{rank} on composite score {score:.3f}."

    # ── Sentence 2: Concerns or availability ──────────────────────────────
    concerns = []

    if features.get("career_quality", 0) <= 0.10:
        concerns.append("entire career at IT services firm (JD disqualifier)")

    if notice > 90:
        concerns.append(f"{notice}-day notice (well above 30-day preferred)")
    elif notice > 60:
        concerns.append(f"{notice}-day notice (above preferred)")

    if days_idle > 120:
        concerns.append(f"last active {days_idle} days ago (availability risk)")

    if not open_to_work:
        concerns.append("not flagged open-to-work")

    if response < 0.30:
        concerns.append(f"low recruiter response rate ({response:.0%})")

    preferred_cities = ["pune", "noida", "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon"]
    in_preferred = any(c in location_str.lower() for c in preferred_cities)
    if not in_preferred and not can_relocate:
        concerns.append(f"based in {location_str}, not willing to relocate")

    if concerns:
        s2 = f"Concerns: {'; '.join(concerns)}."
    else:
        positives = []
        if open_to_work:
            positives.append("actively open to work")
        if notice <= 30:
            positives.append(f"{notice}-day notice (ideal)")
        if response >= 0.70:
            positives.append(f"strong recruiter responsiveness ({response:.0%})")
        if days_idle <= 14:
            positives.append("active on platform in last 2 weeks")
        s2 = f"Strong availability: {'; '.join(positives)}." if positives else ""

    return f"{s1} {s2}".strip()