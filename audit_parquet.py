import pandas as pd
import json

PARQUET_FILE = "candidate_features.parquet"
RAW_FILE = "candidates.jsonl"

def load_raw_profiles():
    print("Loading raw profiles into memory...")
    raw_lookup = {}
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            raw_lookup[cand["candidate_id"]] = cand
    return raw_lookup

def inspect_top_k(df, raw_lookup, sort_column, k=5):
    print(f"\n{'='*50}")
    print(f"🥇 TOP {k} BY: {sort_column.upper()}")
    print(f"{'='*50}")
    
    top_df = df.sort_values(by=sort_column, ascending=False).head(k)
    
    for _, row in top_df.iterrows():
        cid = row['candidate_id']
        score = row[sort_column]
        raw = raw_lookup.get(cid, {})
        
        title = raw.get("profile", {}).get("current_title", "Unknown")
        exp = raw.get("profile", {}).get("years_of_experience", 0)
        summary = raw.get("profile", {}).get("summary", "")[:250] + "..."
        
        print(f"\nID: {cid} | Score: {score:.2f} | Trust: {row['trust_score']} | Builder: {row['builder_ratio']}")
        print(f"Title: {title} ({exp} yrs)")
        print(f"Summary: {summary}")
        print(f"Evidence Found: {row['evidence_terms']}")

def main():
    print("Loading Parquet...")
    df = pd.read_parquet(PARQUET_FILE)
    raw_lookup = load_raw_profiles()
    
    # Create the composite heuristic you asked for
    df['trust_x_retrieval'] = df['trust_score'] * df['retrieval_score']
    
    # Inspect the buckets
    inspect_top_k(df, raw_lookup, "evaluation_score", k=5)
    inspect_top_k(df, raw_lookup, "retrieval_score", k=5)
    inspect_top_k(df, raw_lookup, "vector_db_score", k=5)
    inspect_top_k(df, raw_lookup, "trust_x_retrieval", k=5)

if __name__ == "__main__":
    main()