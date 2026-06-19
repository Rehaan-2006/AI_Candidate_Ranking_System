# honeypot.py
"""
Honeypot detection — returns a multiplier [0.05, 1.0].
Lower = more likely fraudulent/trap profile.
"""

from datetime import datetime


def _education_timeline_valid(candidate):
    """
    Check if education history is plausible.
    Flags: master's degree starting before bachelor's ends.
    """
    edu = candidate.get("education", [])
    degrees = sorted(edu, key=lambda e: e.get("start_year", 9999))
    
    bachelors_end = None
    for deg in degrees:
        level = deg.get("degree", "").lower()
        if any(b in level for b in ["b.tech", "b.e", "b.sc", "be", "btech", "b.com"]):
            bachelors_end = deg.get("end_year")
        elif any(m in level for m in ["m.tech", "m.e", "m.sc", "me", "mtech", "mba", "m.s"]):
            masters_start = deg.get("start_year", 9999)
            if bachelors_end and masters_start < bachelors_end:
                return False  # masters started before bachelors ended
    return True


def detect_honeypot(candidate):
    """
    Returns a multiplier in [0.05, 1.0].
    """
    flags = 0
    skills    = candidate.get("skills", [])
    career    = candidate.get("career_history", [])
    profile   = candidate.get("profile", {})
    signals   = candidate.get("redrob_signals", {})
    assessed  = signals.get("skill_assessment_scores", {})

    # ── Flag 1: Advanced/expert skill with 0 months duration ───────────────
    for skill in skills:
        if skill.get("proficiency") in ["expert", "advanced"] and skill.get("duration_months", 1) == 0:
            flags += 2
            break

    # ── Flag 2: Implausibly many expert skills ──────────────────────────────
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count > 8:
        flags += 2

    # ── Flag 3: Systematic proficiency inflation vs assessment ──────────────
    stuffing = sum(
        1 for s in skills
        if assessed.get(s.get("name", "")) is not None
        and s.get("proficiency") in ["advanced", "expert"]
        and assessed[s["name"]] < 40
    )
    flags += min(stuffing, 3)  # cap at 3

    # ── Flag 4: Career duration impossibility ───────────────────────────────
    # YoE stated >> sum of all job durations
    yoe = float(profile.get("years_of_experience", 0))
    total_months = sum(j.get("duration_months", 0) for j in career)
    if yoe > 2 and total_months < (yoe * 12 * 0.45):
        flags += 2

    # ── Flag 5: Single role spanning >130 months (synthetic data artifact) ──
    for job in career:
        if job.get("duration_months", 0) > 130:
            flags += 1

    # ── Flag 6: Education timeline impossibility ────────────────────────────
    if not _education_timeline_valid(candidate):
        flags += 2

    # ── Flag 7: Non-technical current title + rich AI skill list ───────────
    current_title = profile.get("current_title", "").lower()
    non_tech_titles = ["marketing", "sales", "hr ", "operations manager",
                       "finance", "mechanical engineer", "brand manager"]
    ai_skills = {"ml", "nlp", "llm", "embedding", "deep learning", "rag", "retrieval"}
    has_ai_skills = any(
        kw in s.get("name", "").lower()
        for s in skills for kw in ai_skills
    )
    if has_ai_skills and any(t in current_title for t in non_tech_titles):
        flags += 3

    # ── Multiplier map ──────────────────────────────────────────────────────
    if flags == 0: return 1.00
    if flags == 1: return 0.85
    if flags == 2: return 0.65
    if flags == 3: return 0.40
    if flags == 4: return 0.20
    return 0.05