import json

INPUT_FILE = "candidates.jsonl"

# The expanded Alias Dictionary
AUDIT_TERMS = {
    "Qdrant": ["qdrant"],
    "Milvus": ["milvus"],
    "FAISS": ["faiss", "approximate nearest neighbor", "ann search"],
    "Vector Database": ["vector database", "vector db", "vector store", "vector search infrastructure"],
    "Recommendation Systems": ["recommendation systems", "recommendation engine", "recommender system", "personalization engine", "recsys"],
    "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
    "LlamaIndex": ["llamaindex", "llama_index", "llama index"],
    "LangChain": ["langchain"],
    "NDCG": ["ndcg", "normalized discounted cumulative gain"],
    "MRR": ["mrr", "mean reciprocal rank"]
}

def run_comprehensive_audit():
    print(f"Running Comprehensive Signal Audit on {INPUT_FILE}...")
    
    # Trackers
    # 1. Location Hits
    location_hits = {term: {"skills": 0, "summary": 0, "career": 0} for term in AUDIT_TERMS}
    
    # 2. Evidence Matrix
    matrix = {term: {
        "claimed_supported": 0,    # Listed skill AND wrote about it
        "claimed_unsupported": 0,  # Listed skill BUT never wrote about it (Stuffer)
        "unclaimed_supported": 0   # Didn't list skill BUT wrote about it (Humble Expert)
    } for term in AUDIT_TERMS}
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                cand = json.loads(line)
                
                # Text Blocks
                skills_list = [s.get("name", "").lower() for s in cand.get("skills", [])]
                skills_text = " ".join(skills_list)
                
                summary_text = cand.get("profile", {}).get("summary", "").lower()
                
                career_history = cand.get("career_history", [])
                career_texts = [job.get("description", "").lower() for job in career_history]
                full_career_text = " . ".join(career_texts)
                
                written_evidence_text = f"{summary_text} . {full_career_text}"
                
                for term, aliases in AUDIT_TERMS.items():
                    # Check Claims (Skills array)
                    is_claimed = any(alias in skills_text for alias in aliases)
                    
                    # Check Evidence (Summary & Career)
                    in_summary = any(alias in summary_text for alias in aliases)
                    in_career = any(alias in full_career_text for alias in aliases)
                    is_supported = in_summary or in_career
                    
                    # Log Locations
                    if is_claimed: location_hits[term]["skills"] += 1
                    if in_summary: location_hits[term]["summary"] += 1
                    if in_career: location_hits[term]["career"] += 1
                        
                    # Log Matrix
                    if is_claimed and is_supported:
                        matrix[term]["claimed_supported"] += 1
                    elif is_claimed and not is_supported:
                        matrix[term]["claimed_unsupported"] += 1
                    elif not is_claimed and is_supported:
                        matrix[term]["unclaimed_supported"] += 1

            except json.JSONDecodeError:
                continue

    # Print Table 1: Signal Locations 
    print("\n### Table 1: Signal Location Distribution")
    print("| Term | Skills Hits | Summary Hits | Career Hits |")
    print("|---|---|---|---|")
    for term in AUDIT_TERMS:
        loc = location_hits[term]
        print(f"| {term.ljust(22)} | {str(loc['skills']).ljust(11)} | {str(loc['summary']).ljust(12)} | {str(loc['career']).ljust(11)} |")

    # Print Table 2: The Evidence Matrix
    print("\n### Table 2: Claim vs. Evidence Matrix")
    print("| Term | Claimed + Supported | Claimed + UNSUPPORTED | UNCLAIMED + Supported |")
    print("|---|---|---|---|")
    for term in AUDIT_TERMS:
        m = matrix[term]
        print(f"| {term.ljust(22)} | {str(m['claimed_supported']).ljust(19)} | {str(m['claimed_unsupported']).ljust(21)} | {str(m['unclaimed_supported']).ljust(21)} |")

if __name__ == "__main__":
    run_comprehensive_audit()