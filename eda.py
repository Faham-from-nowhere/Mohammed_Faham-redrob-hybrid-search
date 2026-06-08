import json
import gzip
import os
from collections import defaultdict, Counter

# Configuration
INPUT_FILE = "candidates.jsonl" 
STATS_OUTPUT = "field_stats.json"
ANOMALIES_OUTPUT = "schema_anomalies.json"

def run_eda():
    print(f"Starting highly optimized EDA on {INPUT_FILE}...")
    
    # Trackers
    total_processed = 0
    anomalies = []
    
    # Stats Aggregators
    stats = {
        "skills_duration_types": Counter(),
        "assessment_score_types": Counter(),
        "career_history_lengths": Counter(),
        "experience_years_distribution": Counter(),
        "honeypot_watch": {
            "max_experience_years": 0,
            "max_notice_period": 0,
            "missing_signals": 0
        }
    }

    # Handle both gzipped and raw JSONL seamlessly
    open_func = gzip.open if INPUT_FILE.endswith('.gz') else open
    mode = 'rt' if INPUT_FILE.endswith('.gz') else 'r'

    try:
        with open_func(INPUT_FILE, mode, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    candidate = json.loads(line)
                    total_processed += 1
                    cid = candidate.get("candidate_id", f"UNKNOWN_{total_processed}")
                    
                    # 1. Structural Anomalies Check
                    if "profile" not in candidate or "redrob_signals" not in candidate:
                        anomalies.append({
                            "candidate_id": cid,
                            "issue": "Missing core top-level keys (profile or redrob_signals)"
                        })
                        stats["honeypot_watch"]["missing_signals"] += 1

                    # 2. Track Profile Honeypot Limits
                    profile = candidate.get("profile", {})
                    exp_years = profile.get("years_of_experience")
                    if exp_years is not None:
                        # Round to nearest int for distribution counting
                        stats["experience_years_distribution"][int(exp_years)] += 1
                        if exp_years > stats["honeypot_watch"]["max_experience_years"]:
                            stats["honeypot_watch"]["max_experience_years"] = exp_years

                    # 3. Analyze the Skills Array (Crucial for Evidence Graph)
                    skills = candidate.get("skills", [])
                    for skill in skills:
                        duration = skill.get("duration_months")
                        if duration is None:
                            stats["skills_duration_types"]["null_or_missing"] += 1
                        elif duration == 0:
                            stats["skills_duration_types"]["zero_months"] += 1
                        else:
                            stats["skills_duration_types"]["valid_integer"] += 1

                    # 4. Analyze Assessment Scores
                    signals = candidate.get("redrob_signals", {})
                    assessments = signals.get("skill_assessment_scores")
                    
                    if assessments is None:
                        stats["assessment_score_types"]["strictly_null"] += 1
                    elif isinstance(assessments, dict) and len(assessments) == 0:
                        stats["assessment_score_types"]["empty_dict"] += 1
                    else:
                        stats["assessment_score_types"]["has_scores"] += 1
                        
                    # 5. Track Max Notice Period (Honeypot check)
                    notice_period = signals.get("notice_period_days", 0)
                    if notice_period > stats["honeypot_watch"]["max_notice_period"]:
                        stats["honeypot_watch"]["max_notice_period"] = notice_period

                    # 6. Career History Lengths
                    history = candidate.get("career_history", [])
                    stats["career_history_lengths"][len(history)] += 1

                    # Print progress so you don't stare at a blank terminal
                    if total_processed % 10000 == 0:
                        print(f"Processed {total_processed} candidates...")

                except json.JSONDecodeError:
                    print(f"Warning: JSON decode error on line {total_processed + 1}")
                    
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please check your path.")
        return

    # Finalize and Export
    print(f"\nProcessing complete. Total candidates: {total_processed}")
    
    with open(STATS_OUTPUT, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Exported statistics to {STATS_OUTPUT}")
    
    with open(ANOMALIES_OUTPUT, 'w') as f:
        json.dump(anomalies[:100], f, indent=2) # Only keep top 100 anomalies to save space
    print(f"Exported anomalies (capped at 100) to {ANOMALIES_OUTPUT}")

if __name__ == "__main__":
    run_eda()