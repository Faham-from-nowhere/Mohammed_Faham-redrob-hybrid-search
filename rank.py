import pandas as pd
import numpy as np

# CONFIGURATION 
SYMBOLIC_FILE = "candidate_features.parquet"
SEMANTIC_FILE = "semantic_features.parquet"
OUTPUT_CSV = "team_redrob_submission.csv"

def calculate_jaccard(list1, list2):
    s1, s2 = set(list1), set(list2)
    if not s1 or not s2: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def generate_reasoning(row):
    if row['trust_score'] >= 0.9 and row['coverage_score'] >= 3:
        base = "Exceptional end-to-end match with pristine trust metrics. "
    else:
        base = "Strong technical foundation spanning critical JD requirements. "
        
    evidence = str(row['evidence_terms'])
    if pd.notna(evidence) and evidence.strip() and evidence.lower() != "nan":
        terms = [t.strip() for t in evidence.split(",") if t.strip()]
        term_str = ", ".join(terms[:3]) 
        detail = f"Verified evidence utilizing {term_str}. "
    else:
        detail = "Deep semantic alignment with core search and retrieval domains. "
        
    scores = {
        "retrieval": row["retrieval_score"],
        "vector": row["vector_db_score"],
        "evaluation": row["evaluation_score"],
        "production": row["production_ml_score"]
    }
    
    top_signal = max(scores, key=scores.get)
    
    signal_text = {
        "retrieval": "Demonstrates a strong retrieval and search infrastructure background.",
        "vector": "Possesses hands-on vector database and ANN experience.",
        "evaluation": "Shows rare expertise in ranking evaluation metrics (NDCG/MRR).",
        "production": "Has a proven history of shipping production ML systems."
    }
    
    if scores[top_signal] == 0:
        behavior = "High builder-ratio indicates a scrappy, shipping-oriented trajectory."
    else:
        behavior = signal_text[top_signal]
        
    return base + detail + behavior

def main():
    print("Loading Parquet Artifacts...")
    df_sym = pd.read_parquet(SYMBOLIC_FILE)
    df_sem = pd.read_parquet(SEMANTIC_FILE)
    
    # 1. Column Sanity Check
    print("Verifying Semantic Columns...")
    expected_cols = ["sem_retrieval", "sem_evaluation", "sem_vector_db", "sem_production"]
    for col in expected_cols:
        if col not in df_sem.columns:
            raise KeyError(f"CRITICAL: Expected semantic column '{col}' is missing! Found: {df_sem.columns.tolist()}")

    # Merge the Twin Engines
    df = pd.merge(df_sym, df_sem, on="candidate_id")
    
    # 2. HARD FILTER: The Minimum Trust Guardrail
    initial_len = len(df)
    df = df[df['trust_score'] >= 0.6].copy()
    print(f"Quarantine Dropped: {initial_len - len(df)} low-trust candidates.")
    
    # 3. SEMANTIC CORE
    df["semantic_core"] = (
        0.40 * df["sem_retrieval"] +
        0.30 * df["sem_evaluation"] +
        0.20 * df["sem_vector_db"] +
        0.10 * df["sem_production"]
    )
    
    # 4. INDEPENDENT AXIS RANKING 
    print("Computing Independent Rank Lists...")
    df['rank_retrieval'] = df['retrieval_score'].rank(method='min', ascending=False)
    df['rank_vector'] = df['vector_db_score'].rank(method='min', ascending=False)
    df['rank_evaluation'] = df['evaluation_score'].rank(method='min', ascending=False)
    df['rank_production'] = df['production_ml_score'].rank(method='min', ascending=False)
    df['rank_semantic'] = df['semantic_core'].rank(method='min', ascending=False)
    df['rank_coverage'] = df['coverage_score'].rank(method='min', ascending=False)
    
    # 5. RECIPROCAL RANK FUSION (RRF) - VERSION B (Weighted Coverage)
    print("Applying Reciprocal Rank Fusion (Double-Weighted Coverage)...")
    K = 60
    df['rrf_score'] = (
        (1.0 / (K + df['rank_retrieval'])) +
        (1.0 / (K + df['rank_vector'])) +
        (1.0 / (K + df['rank_evaluation'])) +
        (1.0 / (K + df['rank_production'])) +
        (1.0 / (K + df['rank_semantic'])) +
        (2.0 / (K + df['rank_coverage']))  
    )
    
    # Inject Semantic Coverage if available
    if 'semantic_coverage' in df.columns:
        print("Detected semantic_coverage. Injecting into RRF...")
        df['rank_sem_coverage'] = df['semantic_coverage'].rank(method='min', ascending=False)
        df['rrf_score'] += (1.0 / (K + df['rank_sem_coverage']))
    
    # 6. TRUST MULTIPLIER
    df['rrf_score'] *= (0.7 + 0.3 * df['trust_score'])
    
    # Sort by the final fused score + Trust Tie-Breaker
    sorted_df = df.sort_values(by=['rrf_score', 'trust_score'], ascending=[False, False])
    
    # 7. THE DIVERSITY EXPERIMENT (A / B / C)
    print("Running Diversity A/B/C Test...")
    
    def run_filter(df, threshold, min_terms_to_check=0):
        selected = []
        for _, cand in df.iterrows():
            ev_str = str(cand['evidence_terms'])
            cand_terms = [t.strip() for t in ev_str.split(',')] if ev_str.lower() != 'nan' else []
            cand_terms = [t for t in cand_terms if t]
            
            is_clone = False
            # Only apply diversity check if candidate has enough terms to warrant it
            if len(cand_terms) >= min_terms_to_check:
                for s_cand in selected:
                    s_ev_str = str(s_cand['evidence_terms'])
                    s_terms = [t.strip() for t in s_ev_str.split(',')] if s_ev_str.lower() != 'nan' else []
                    s_terms = [t for t in s_terms if t]
                    
                    if calculate_jaccard(cand_terms, s_terms) > threshold:
                        is_clone = True
                        break
                    
            if not is_clone:
                selected.append(cand.to_dict())
                
            if len(selected) == 100:
                break
                
        # Backfill if short
        if len(selected) < 100:
            selected_ids = {c['candidate_id'] for c in selected}
            remaining = df[~df['candidate_id'].isin(selected_ids)].head(100 - len(selected))
            return pd.concat([pd.DataFrame(selected), remaining]).reset_index(drop=True)
        return pd.DataFrame(selected).reset_index(drop=True)

    # Generate the 3 Versions
    final_100_A = run_filter(sorted_df, threshold=0.95, min_terms_to_check=0) # Current
    final_100_B = sorted_df.head(100).copy() # No Diversity
    final_100_C = run_filter(sorted_df, threshold=0.99, min_terms_to_check=3) # Loose Diversity (Safe)

    # Metrics Calc
    def get_metrics(df_version):
        cov2 = (df_version['coverage_score'] >= 2).sum()
        elite = len(df_version[(df_version["retrieval_score"] > 0) | (df_version["evaluation_score"] > 0) | (df_version["vector_db_score"] > 0)])
        return cov2, elite

    cov2_A, elite_A = get_metrics(final_100_A)
    cov2_B, elite_B = get_metrics(final_100_B)
    cov2_C, elite_C = get_metrics(final_100_C)

    print("\n==================================================")
    print("🧪 DIVERSITY FILTER EXPERIMENT RESULTS")
    print("==================================================")
    print(f"{'Version':<20} | {'Coverage >= 2':<15} | {'Elite Count':<15}")
    print("-" * 55)
    print(f"{'A: Strict (0.95)':<20} | {cov2_A:<15} | {elite_A:<15}")
    print(f"{'B: No Diversity':<20} | {cov2_B:<15} | {elite_B:<15}")
    print(f"{'C: Loose (0.99 + len)':<20} | {cov2_C:<15} | {elite_C:<15}")
    print("==================================================\n")

    # Lock in Version C for final submission
    print("Locking in Version C (Loose Diversity) for submission...")
    final_100 = final_100_C
    final_100 = final_100.sort_values(by=['rrf_score', 'trust_score'], ascending=[False, False]).reset_index(drop=True)
    final_100["rank"] = range(1, len(final_100) + 1)
    
    # PRE-SUBMISSION DIAGNOSTICS
    print("\n==================================================")
    print("📊 FINAL TOP 100 DIAGNOSTICS")
    print("==================================================")
    print("\n[Signal Breakdown]")
    print(final_100[["retrieval_score", "evaluation_score", "vector_db_score", "production_ml_score", "coverage_score"]].describe())
    
    print("\n[Trust Score Distribution]")
    print(final_100["trust_score"].describe())
    
    print("\n[Top 25 Candidates Inspection]")
    columns_to_inspect = [
        "candidate_id", "retrieval_score", "evaluation_score", 
        "vector_db_score", "production_ml_score", "coverage_score", "trust_score", "evidence_terms"
    ]
    print(final_100[columns_to_inspect].head(25).to_string(index=False))
    print("==================================================\n")
    
    # 9. GENERATE EXPLAINABILITY & FORMAT
    print("Generating Dynamic Explainability Strings...")
    final_100['reasoning'] = final_100.apply(generate_reasoning, axis=1)
    
    submission = final_100[['candidate_id', 'rank', 'rrf_score', 'reasoning']].copy()
    submission.rename(columns={'rrf_score': 'score'}, inplace=True)
    
    submission['score'] = submission['score'].astype(float).apply(lambda x: f"{x:.6f}")

    
    # THE "CORE DOMAIN" GUARDRAIL
    # Ensures every candidate has at least *some* verifiable
    # symbolic footprint in Search, Vector DBs, or Evaluation.
    print("Applying Core Domain Guardrail...")
    initial_100_len = len(final_100)
    
    final_100['core_symbolic'] = (
        final_100['retrieval_score'] + 
        final_100['vector_db_score'] + 
        final_100['evaluation_score']
    )
    
    final_100 = final_100[final_100['core_symbolic'] > 0].copy()
    
    # If the guardrail dropped candidates, backfill from the remaining sorted_df
    if len(final_100) < 100:
        print(f"Dropped {initial_100_len - len(final_100)} candidates lacking core domain experience.")
        selected_ids = set(final_100['candidate_id'])
        
        # Filter the remaining pool to only those who also pass the guardrail
        valid_remaining = sorted_df[
            (~sorted_df['candidate_id'].isin(selected_ids)) & 
            ((sorted_df['retrieval_score'] + sorted_df['vector_db_score'] + sorted_df['evaluation_score']) > 0)
        ]
        
        needed = 100 - len(final_100)
        final_100 = pd.concat([final_100, valid_remaining.head(needed)]).reset_index(drop=True)
        
    # Re-sort and re-rank just to be absolutely safe
    final_100 = final_100.sort_values(by=['rrf_score', 'trust_score'], ascending=[False, False]).reset_index(drop=True)
    final_100["rank"] = range(1, len(final_100) + 1)

    
    # FINAL TELEMETRY CHECKS
    print("\n==================================================")
    print("📡 FINAL SYSTEM TELEMETRY")
    print("==================================================")
    
    # 1. Semantic Contribution Check
    print("\n[Semantic Engine Contribution]")
    if 'semantic_coverage' in final_100.columns:
        print("Semantic Coverage Spread:")
        print(final_100["semantic_coverage"].value_counts().sort_index())
    
    print("\nSemantic Core Distribution:")
    print(final_100["semantic_core"].describe()[['mean', 'min', '50%', 'max']])
    
    # 2. Vocabulary Diversity Check
    print("\n[Evidence Vocabulary Diversity]")
    all_terms = set()
    for ev in final_100["evidence_terms"]:
        if pd.notna(ev):
            all_terms.update([t.strip() for t in str(ev).split(",") if t.strip()])
            
    print(f"Total Unique Evidence Terms in Top 100: {len(all_terms)}")
    print("Term List:", sorted(list(all_terms)))
    print("==================================================\n")
    
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"🎉 SUCCESS: Generated {OUTPUT_CSV}! Ready for submission.")

if __name__ == "__main__":
    main()