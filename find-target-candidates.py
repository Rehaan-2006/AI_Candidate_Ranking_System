import json
import os

def get_current_role_info(candidate):
    """Safely extract the current title and company from the top-level career_history."""
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])  
    
    if career and isinstance(career, list) and len(career) > 0:
        current = career[0]
        return {
            "title": current.get("title", profile.get("current_title", profile.get("headline", ""))),
            "company": current.get("company", "Unknown Company")
        }
    
    return {
        "title": profile.get("current_title", profile.get("headline", "")),
        "company": "Unknown Company"
    }

def is_in_india(profile):
    """Check if the candidate is located in India."""
    loc = str(profile.get("location", "")).lower()
    india_keywords = ["india", "bangalore", "bengaluru", "pune", "noida", "hyderabad", "mumbai", "delhi", "gurgaon", "chennai", "kolkata"]
    return any(kw in loc for kw in india_keywords)

def guess_company_type(company_name):
    """Simple heuristic to flag known IT service giants."""
    services_giants = ["tcs", "tata consultancy", "wipro", "infosys", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree"]
    name_lower = company_name.lower()
    
    if any(giant in name_lower for giant in services_giants):
        return "IT Services"
    if company_name == "Unknown Company":
        return "Unknown"
    return "Likely Product/Startup"

def find_target_candidates(file_path="candidates.jsonl", limit=4):
    found_candidates = []
    
    print(f"Scanning {file_path} for target AI candidates...\n")
    
    try:
        with open(file_path, "rt", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                    
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                signals = candidate.get("redrob_signals", {})
                profile = candidate.get("profile", {})
                
                # 1. Behavioral Signals
                github_score = signals.get("github_activity_score", -1)
                interview_rate = signals.get("interview_completion_rate", 0.0)
                
                if github_score <= 70 or interview_rate <= 0.7:
                    continue
                    
                # 2. Location Check
                if not is_in_india(profile):
                    continue
                    
                # 3. Years of Experience Check
                yoe = profile.get("years_of_experience", profile.get("total_experience_years", 0))
                try:
                    yoe = float(yoe)
                except (ValueError, TypeError):
                    yoe = 0.0
                    
                if not (4 <= yoe <= 12):
                    continue

                # 4. Title/Domain Check (Now passing the full candidate object)
                role_info = get_current_role_info(candidate)
                title_lower = role_info["title"].lower()
                
                target_keywords = ["ai", "ml", "machine learning", "nlp", "search", "ranking"]
                if not any(kw in title_lower for kw in target_keywords):
                    continue
                    
                # 5. Capture Match
                company_type = guess_company_type(role_info["company"])
                
                candidate_summary = {
                    "id": candidate.get("candidate_id"),
                    "name": profile.get("anonymized_name", "Unknown"),
                    "title": role_info["title"],
                    "company": role_info["company"],
                    "company_type": company_type,
                    "yoe": yoe,
                    "github_score": github_score,
                    "interview_rate": interview_rate,
                    "location": profile.get("location", "India (Assumed)")
                }
                
                found_candidates.append((candidate, candidate_summary))
                
                print(f"✅ Match Found: {candidate_summary['name']} ({candidate_summary['id']})")
                print(f"   Role: {candidate_summary['title']} @ {candidate_summary['company']} [{candidate_summary['company_type']}]")
                print(f"   Stats: {candidate_summary['yoe']} YoE | GitHub: {candidate_summary['github_score']} | Interview Rate: {candidate_summary['interview_rate']}")
                print("-" * 60)
                
                if len(found_candidates) >= limit:
                    break
                    
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{file_path}'.")
        print("Please ensure your file is named exactly 'candidates.jsonl' and is in the same folder as this script.")
        return []

    return found_candidates

if __name__ == "__main__":
    results = find_target_candidates(limit=4)
    
    if results:
        full_profiles = [r[0] for r in results]
        output_file = "target_candidates.json"
        
        with open(output_file, "w", encoding="utf-8") as out_file:
            json.dump(full_profiles, out_file, indent=4)
            
        print(f"\nSaved full JSON profiles to '{output_file}'.")
        print("Please paste the contents of that file here so we can architect the embedding text builder!")
    else:
        print("\nNo candidates found matching all criteria. We may need to loosen the filters slightly.")