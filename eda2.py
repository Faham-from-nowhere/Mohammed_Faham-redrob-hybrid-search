import json
import gzip
import random
from collections import Counter

# Configuration
INPUT_FILE = "candidates.jsonl" 
OUTPUT_STATS = "semantic_stats.json"
OUTPUT_SAMPLES = "random_samples.json"

# The specific terms we want to baseline before building the alias engine
JD_TARGET_TERMS = [
    "milvus", "qdrant", "faiss", "pinecone", "vector database", 
    "semantic search", "rag", "dense retrieval", "ranking", 
    "ndcg", "mrr", "a/b testing", "ab testing"
]

def run_semantic_eda():
    print(f"Starting Semantic EDA on {INPUT_FILE}...")
    
    total_processed = 0
    
    # Trackers
    stats = {
        "top_skills": Counter(),
        "top_titles": Counter(),
        "retrieval_term_frequency": Counter(),
        "description_word_counts": {
            "under_20": 0,
            "20_to_50": 0,
            "50_to_100": 0,
            "over_100": 0,
            "total_descriptions_parsed": 0
        }
    }
    
    random_samples = []

    open_func = gzip.open if INPUT_FILE.endswith('.gz') else open
    mode = 'rt' if INPUT_FILE.endswith('.gz') else 'r'

    with open_func(INPUT_FILE, mode, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                candidate = json.loads(line)
                total_processed += 1
                
                #  1. Skill Frequency
                for skill in candidate.get("skills", []):
                    name = skill.get("name", "").strip().lower()
                    if name:
                        stats["top_skills"][name] += 1
                
                # 2. Title Frequency
                profile_title = candidate.get("profile", {}).get("current_title", "").strip().lower()
                if profile_title:
                    stats["top_titles"][profile_title] += 1
                
                career = candidate.get("career_history", [])
                full_text_blocks = [candidate.get("profile", {}).get("summary", "").lower()]
                
                for job in career:
                    job_title = job.get("title", "").strip().lower()
                    if job_title:
                        stats["top_titles"][job_title] += 1
                        
                    # 4. Description Richness 
                    desc = job.get("description", "")
                    if desc:
                        full_text_blocks.append(desc.lower())
                        word_count = len(desc.split())
                        stats["description_word_counts"]["total_descriptions_parsed"] += 1
                        
                        if word_count < 20: stats["description_word_counts"]["under_20"] += 1
                        elif word_count < 50: stats["description_word_counts"]["20_to_50"] += 1
                        elif word_count < 100: stats["description_word_counts"]["50_to_100"] += 1
                        else: stats["description_word_counts"]["over_100"] += 1

                # 3. Retrieval-Term Frequency 
                full_text = " ".join(full_text_blocks)
                for term in JD_TARGET_TERMS:
                    if term in full_text:
                        stats["retrieval_term_frequency"][term] += 1

                # 5. Random Candidate Inspection 
                # Reservoir sampling: keep exactly 5 candidates randomly distributed across the 100K
                if len(random_samples) < 5:
                    random_samples.append(candidate)
                else:
                    j = random.randint(0, total_processed - 1)
                    if j < 5:
                        random_samples[j] = candidate

                if total_processed % 10000 == 0:
                    print(f"Processed {total_processed} candidates...")

            except json.JSONDecodeError:
                continue

    # Format the output so I don't dump a 50MB dictionary
    print("\nProcessing complete. Formatting outputs...")
    final_stats = {
        "description_richness": stats["description_word_counts"],
        "retrieval_term_frequency": dict(stats["retrieval_term_frequency"].most_common()),
        "top_50_skills": dict(stats["top_skills"].most_common(50)),
        "top_50_titles": dict(stats["top_titles"].most_common(50))
    }

    with open(OUTPUT_STATS, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    with open(OUTPUT_SAMPLES, 'w') as f:
        json.dump(random_samples, f, indent=2)

    print(f"Exported stats to {OUTPUT_STATS}")
    print(f"Exported 5 random samples to {OUTPUT_SAMPLES}")

if __name__ == "__main__":
    run_semantic_eda()