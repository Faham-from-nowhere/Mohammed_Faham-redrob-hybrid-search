import pandas as pd

PARQUET_FILE = "candidate_features.parquet"

def main():
    print("Loading Parquet for Advanced Diagnostics...\n")
    df = pd.read_parquet(PARQUET_FILE)
    
    #1. COVERAGE DISTRIBUTION 
    print("="*40)
    print("🔍 COVERAGE SCORE DISTRIBUTION")
    print("="*40)
    dist = df["coverage_score"].value_counts().sort_index()
    for score, count in dist.items():
        print(f"Coverage {score}: {count} candidates")
        
    # 2. OVERLAP ANALYSIS
    print("\n" + "="*40)
    print("🧬 TOP 100 OVERLAP MATRIX")
    print("="*40)
    
    # Get top 100 IDs for each metric
    top_eval = set(df.nlargest(100, "evaluation_score")["candidate_id"])
    top_retrieval = set(df.nlargest(100, "retrieval_score")["candidate_id"])
    top_vector = set(df.nlargest(100, "vector_db_score")["candidate_id"])
    top_symbolic = set(df.nlargest(100, "baseline_symbolic_score")["candidate_id"])
    
    # Calculate overlaps
    eval_retrieval = len(top_eval & top_retrieval)
    eval_symbolic = len(top_eval & top_symbolic)
    retrieval_symbolic = len(top_retrieval & top_symbolic)
    vector_symbolic = len(top_vector & top_symbolic)
    
    print(f"Eval & Retrieval Overlap:       {eval_retrieval}%")
    print(f"Eval & Baseline Symbolic:       {eval_symbolic}%")
    print(f"Retrieval & Baseline Symbolic:  {retrieval_symbolic}%")
    print(f"Vector DB & Baseline Symbolic:  {vector_symbolic}%")
    
    # The Elite Triple Threat (Retrieval + Eval + Vector)
    triple_threat = top_eval & top_retrieval & top_vector
    print(f"\nCandidates in Top 100 for ALL THREE (Retrieval/Eval/Vector): {len(triple_threat)}")

if __name__ == "__main__":
    main()