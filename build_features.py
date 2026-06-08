import json
import math
import pandas as pd
from flashtext import KeywordProcessor

# 1. THE RELEVANCE MATRIX (JD ALIGNMENT)
JD_WEIGHTS = {
    "ndcg": 2.0, "mrr": 2.0, "map": 2.0, "mean average precision": 2.0, 
    "offline evaluation": 2.0, "learning to rank": 2.0, "ltr": 2.0,
    "ranking metrics": 2.0, "ranking evaluation": 2.0,
    
    "semantic search": 1.5, "dense retrieval": 1.5, "hybrid search": 1.5,
    "retrieval pipeline": 1.5, "candidate matching": 1.5, "information retrieval": 1.5,
    "faiss": 1.5, "milvus": 1.2, "qdrant": 1.2, "pinecone": 1.2, "weaviate": 1.2,
    "ann search": 1.5, "a/b testing": 1.5, "ab testing": 1.5,
    
    "rag": 0.5, "retrieval augmented generation": 0.5, "langchain": 0.3,
    "vector database": 0.5, "vector store": 0.5, "vector db": 0.5
}

#  2. CLUSTERS & VERBS
CLUSTERS = {
    "retrieval": ["semantic search", "dense retrieval", "hybrid search", "retrieval pipeline", "candidate matching", "information retrieval", "rag", "retrieval augmented generation"],
    "vector_db": ["vector database", "vector store", "vector db", "milvus", "qdrant", "pinecone", "weaviate", "faiss", "ann search"],
    "evaluation": ["ndcg", "mrr", "map", "mean average precision", "a/b testing", "ab testing", "offline evaluation", "learning to rank", "ltr", "ranking metrics", "ranking evaluation"],
    "production_ml": ["production ml", "mlops", "model serving", "latency", "embedding drift", "ci/cd", "docker", "kubernetes", "fastapi"],
    "startup_dna": ["zero to one", "founding engineer", "greenfield", "end to end", "owned the product", "startup", "fast paced"]
}

VERBS = {
    "builder": ["architected", "built", "deployed", "shipped", "designed", "implemented", "owned", "scaled", "launched", "fine-tuned", "finetuned"],
    "support": ["used", "assisted", "supported", "helped", "participated", "familiar with", "explored", "researched", "responsible for", "collaborated", "contributed", "maintained", "involved in"]
}

# 3. HASH MAPS & PROCESSORS
TERM_TO_CLUSTER = {alias: cluster for cluster, aliases in CLUSTERS.items() for alias in aliases}

exact_processor = KeywordProcessor()
for aliases in CLUSTERS.values():
    for a in aliases: exact_processor.add_keyword(a, a)

verb_processor = KeywordProcessor()
for v, aliases in VERBS.items():
    for a in aliases: verb_processor.add_keyword(a, v)


# 4. TRUST SCORE LOGIC 
def calculate_trust_score(candidate, exp_years, summary, full_career_text):
    trust = 1.0
    if exp_years > 30 or candidate.get("redrob_signals", {}).get("notice_period_days", 0) > 180: trust -= 0.5
    
    current_title = candidate.get("profile", {}).get("current_title", "").lower()
    if any(p in summary and p not in current_title for p in ["marketing manager", "hr manager", "accountant", "civil engineer", "graphic designer", "sales"]): trust -= 0.3
    if any(p in summary for p in ["experimented with chatgpt", "prompt engineering", "asked chatgpt", "familiar with ai"]): trust -= 0.2
    
    total_months = sum(job.get("duration_months", 0) for job in candidate.get("career_history", []) if job.get("duration_months"))
    if exp_years > 0 and (total_months / 12.0) < (exp_years * 0.4): trust -= 0.2
    
    skills_listed = [s.get("name", "").lower() for s in candidate.get("skills", [])]
    if skills_listed:
        evidence_text = f"{summary} . {full_career_text}"
        supported = sum(1 for s in skills_listed if s in evidence_text)
        ratio = supported / len(skills_listed)
        
        # Softened penalties for high-skill engineers who just don't write about everything
        if ratio < 0.1: trust -= 0.2 
        elif ratio < 0.3: trust -= 0.1
            
    return max(0.0, round(trust, 2))


# 5. CORE EXTRACTION
def extract_features(candidate, rarity_weights):
    cid = candidate["candidate_id"]
    exp_years = candidate.get("profile", {}).get("years_of_experience", 0)
    
    summary = candidate.get("profile", {}).get("summary", "").lower()
    career = " . ".join([j.get("description", "").lower() for j in candidate.get("career_history", [])])
    
    trust_score = calculate_trust_score(candidate, exp_years, summary, career)
    
    verb_matches = verb_processor.extract_keywords(f"{summary} . {career}")
    b_count, s_count = verb_matches.count("builder"), verb_matches.count("support")
    builder_ratio = (b_count + 1) / (b_count + s_count + 2)
    
    cluster_scores = { "retrieval_score": 0.0, "vector_db_score": 0.0, "evaluation_score": 0.0, "production_ml_score": 0.0, "startup_dna_score": 0.0 }
    evidence_terms_found = set()

    career_terms = set(exact_processor.extract_keywords(career))
    for exact_term in career_terms:
        cluster = TERM_TO_CLUSTER[exact_term]
        r_weight = min(8.0, rarity_weights.get(exact_term, 1.0))
        jd_weight = JD_WEIGHTS.get(exact_term, 1.0) 
        cluster_scores[f"{cluster}_score"] += (1.0 * r_weight * jd_weight)
        evidence_terms_found.add(exact_term)
        
    summary_terms = set(exact_processor.extract_keywords(summary))
    for exact_term in summary_terms:
        cluster = TERM_TO_CLUSTER[exact_term]
        r_weight = min(8.0, rarity_weights.get(exact_term, 1.0))
        jd_weight = JD_WEIGHTS.get(exact_term, 1.0) 
        cluster_scores[f"{cluster}_score"] += (0.4 * r_weight * jd_weight)
        evidence_terms_found.add(exact_term)
        
    coverage_score = sum(1 for score in cluster_scores.values() if score > 0)
        
    signals = candidate.get("redrob_signals", {})
    search_app = signals.get("search_appearance_30d", 0)
    saved = signals.get("saved_by_recruiters_30d", 0)
    market_score = math.log1p(search_app) + (2 * math.log1p(saved))

    return {
        "candidate_id": cid,
        "trust_score": trust_score,
        "builder_ratio": round(builder_ratio, 3),
        "market_score": round(market_score, 3),
        "coverage_score": coverage_score,
        "evidence_terms": ",".join(list(evidence_terms_found)),
        **{k: round(v, 3) for k, v in cluster_scores.items()}
    }

def main():
    try:
        with open("term_rarity_weights.json", "r") as f:
            rarity_weights = json.load(f)
    except FileNotFoundError:
        print("Warning: term_rarity_weights.json not found. Using baseline IDF=1.0")
        rarity_weights = {}

    print("Extracting features from 100K candidates...")
    final_data = []
    
    with open("candidates.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip(): 
                final_data.append(extract_features(json.loads(line), rarity_weights))
            if (i+1) % 20000 == 0: 
                print(f"{i+1} processed...")
                
    df = pd.DataFrame(final_data)
    
    print("Normalizing distributions and generating baseline symbolic score...")
    df['market_score_norm'] = df['market_score'] / df['market_score'].max() if df['market_score'].max() > 0 else 0
    df['coverage_score_norm'] = df['coverage_score'] / 5.0
    
    df['baseline_symbolic_score'] = (
        (0.30 * df['trust_score']) +
        (0.25 * df['builder_ratio']) +
        (0.15 * df['market_score_norm']) +
        (0.30 * df['coverage_score_norm'])
    )
    
    df.to_parquet("candidate_features.parquet", index=False)
    print("SUCCESS: Final production-ready candidate_features.parquet generated.")

if __name__ == "__main__":
    main()