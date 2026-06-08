import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# CONFIGURATION 
INPUT_FILE = "candidates.jsonl"
OUTPUT_PARQUET = "semantic_features.parquet"
OUTPUT_FAISS = "faiss.index"
MODEL_NAME = "all-MiniLM-L6-v2"

# The 5 Independent JD Query Vectors
JD_AXES = {
    "sem_retrieval": "semantic search, dense retrieval, candidate matching, hybrid search, retrieval pipeline",
    "sem_vector_db": "scaling vector databases, infrastructure, Qdrant, Milvus, FAISS, approximate nearest neighbor",
    "sem_evaluation": "ranking evaluation metrics, NDCG, mean reciprocal rank, learning to rank, A/B testing",
    "sem_production": "MLOps, deploying machine learning models to production, Kubernetes, latency optimization",
    "sem_startup": "zero to one founding engineer, rapid prototyping, end-to-end ownership in a fast-paced startup"
}

def main():
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    
    # 1. Embed the 5 Queries
    print("Embedding JD Axes...")
    axis_names = list(JD_AXES.keys())
    axis_texts = list(JD_AXES.values())
    query_vectors = model.encode(axis_texts, normalize_embeddings=True)
    
    # Setup FAISS
    d = query_vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    
    candidates_data = []
    texts = []
    cids = []
    batch_size = 2000
    
    print("Streaming candidates for Phase 1A...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            cand = json.loads(line)
            
            career = " . ".join([j.get("description", "") for j in cand.get("career_history", [])])
            summary = cand.get("profile", {}).get("summary", "")
            full_text = f"{summary} {career}"
            
            texts.append(full_text)
            cids.append(cand["candidate_id"])
            
            if len(texts) >= batch_size:
                cand_vectors = model.encode(texts, normalize_embeddings=True)
                index.add(np.array(cand_vectors))
                similarities = np.dot(cand_vectors, query_vectors.T)
                
                for j in range(len(cids)):
                    row_data = {"candidate_id": cids[j]}
                    for k, axis_name in enumerate(axis_names):
                        row_data[axis_name] = float(similarities[j][k])
                    candidates_data.append(row_data)
                    
                texts = []
                cids = []
                print(f"Processed {i + 1} profiles...")
                
        # Catch the last batch
        if texts:
            cand_vectors = model.encode(texts, normalize_embeddings=True)
            index.add(np.array(cand_vectors))
            similarities = np.dot(cand_vectors, query_vectors.T)
            for j in range(len(cids)):
                row_data = {"candidate_id": cids[j]}
                for k, axis_name in enumerate(axis_names):
                    row_data[axis_name] = float(similarities[j][k])
                candidates_data.append(row_data)

    # Save FAISS Index immediately
    faiss.write_index(index, OUTPUT_FAISS)
    
    # DATASET-LEVEL SEMANTIC NORMALIZATION 
    print("\nCalculating Semantic Coverage and Core Metrics via Pandas...")
    df = pd.DataFrame(candidates_data)
    
    # 1. Dynamic Thresholds (The 95th Percentile Rule)
    print("Computing 95th percentile thresholds per axis...")
    thresholds = {}
    for axis in axis_names:
        thresholds[axis] = df[axis].quantile(0.95)
        print(f"  {axis} Top 5% Threshold: {thresholds[axis]:.4f}")
        
    # 2. Compute Semantic Coverage
    def calculate_coverage(row):
        return sum(1 for axis in axis_names if row[axis] >= thresholds[axis])
        
    df['semantic_coverage'] = df.apply(calculate_coverage, axis=1)
    
    # 3. Compute Semantic Core (Weighted towards the JD's primary needs)
    # 40% Retrieval, 30% Vector DB, 30% Evaluation
    df['semantic_overall'] = (
        (df['sem_retrieval'] * 0.4) + 
        (df['sem_vector_db'] * 0.3) + 
        (df['sem_evaluation'] * 0.3)
    )

    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSUCCESS: Generated {OUTPUT_FAISS} and mathematically sound {OUTPUT_PARQUET}!")

if __name__ == "__main__":
    main()