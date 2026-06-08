import json
import math
from flashtext import KeywordProcessor

INPUT_FILE = "candidates.jsonl"
OUTPUT_WEIGHTS = "term_rarity_weights.json"
TOTAL_CANDIDATES = 100000

# The exact clusters from Phase 2
CLUSTERS = {
    "retrieval": ["semantic search", "dense retrieval", "hybrid search", "retrieval pipeline", "candidate matching", "information retrieval", "rag"],
    "vector_db": ["vector database", "vector db", "vector store", "milvus", "qdrant", "pinecone", "weaviate", "faiss", "ann search"],
    "evaluation": ["ndcg", "mrr", "map", "mean average precision", "a/b testing", "ab testing", "offline evaluation", "learning to rank", "ltr", "ranking metrics", "ranking evaluation"],
    "production_ml": ["production ml", "mlops", "model serving", "latency", "embedding drift", "ci/cd", "docker", "kubernetes", "fastapi"],
    "startup_dna": ["zero to one", "founding engineer", "greenfield", "end to end", "owned the product", "startup", "fast paced"]
}

def generate_weights():
    print("Scanning dataset to calculate Signal Rarity...")
    processor = KeywordProcessor()
    
    # We map terms to themselves so FlashText extracts the exact words found
    for cluster, aliases in CLUSTERS.items():
        for alias in aliases:
            processor.add_keyword(alias, alias) 
            
    doc_frequencies = {alias: 0 for aliases in CLUSTERS.values() for alias in aliases}
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                cand = json.loads(line)
                
                # We calculate rarity based ONLY on written evidence (Summary + Career)
                career = " . ".join([j.get("description", "").lower() for j in cand.get("career_history", [])])
                summary = cand.get("profile", {}).get("summary", "").lower()
                full_text = f"{summary} . {career}"
                
                # Get unique terms in this document
                found_terms = set(processor.extract_keywords(full_text))
                for term in found_terms:
                    doc_frequencies[term] += 1
                    
            except json.JSONDecodeError:
                continue

    # Calculate IDF Weights
    weights = {}
    for term, freq in doc_frequencies.items():
        # IDF formula with smoothing
        weight = math.log(TOTAL_CANDIDATES / (freq + 1))
        # Cap minimum weight at 0.5 so commodity terms still give some points
        weights[term] = max(0.5, round(weight, 3))
        
    # Save the lookup table
    with open(OUTPUT_WEIGHTS, "w") as f:
        json.dump(weights, f, indent=2)
        
    print(f"Exported rarity weights to {OUTPUT_WEIGHTS}!")
    
    # Print a few examples to verify
    print("\nSample Weights:")
    for term in ["rag", "ndcg", "milvus", "semantic search", "faiss"]:
        if term in weights: print(f"{term.ljust(20)}: {weights[term]}")

if __name__ == "__main__":
    generate_weights()