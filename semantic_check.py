import pandas as pd

def main():
    print("Loading Semantic Features...")
    df_sem = pd.read_parquet("semantic_features.parquet")

    print("\n==================================================")
    print("📊 SEMANTIC PIPELINE DIAGNOSTICS")
    print("==================================================")

    print("\n1. Semantic Coverage Distribution:")
    print(df_sem["semantic_coverage"].value_counts().sort_index())

    print("\n2. Axis Summary Statistics:")
    for col in ["sem_retrieval", "sem_vector_db", "sem_evaluation", "sem_production", "sem_startup"]:
        print(f"\n--- {col} ---")
        # Showing just count, mean, std, min, 50%, 95%, max for a cleaner readout
        desc = df_sem[col].describe(percentiles=[.50, .95])
        print(desc[['count', 'mean', 'std', 'min', '50%', '95%', 'max']])

    print("\n3. Top 20 Semantic Candidates (By Core JD Alignment):")
    # Recreating the semantic_core calculation from rank.py for accurate top 20 sorting
    df_sem["semantic_core"] = (
        0.40 * df_sem["sem_retrieval"] +
        0.30 * df_sem["sem_evaluation"] +
        0.20 * df_sem["sem_vector_db"] +
        0.10 * df_sem["sem_production"]
    )
    
    cols_to_show = [
        "candidate_id", "semantic_coverage", "semantic_core", 
        "sem_retrieval", "sem_vector_db", "sem_evaluation", "sem_production"
    ]
    print(df_sem.sort_values("semantic_core", ascending=False).head(20)[cols_to_show].to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    main()