import pandas as pd

df = pd.read_csv("team_redrob_submission.csv")

# We need to re-merge with features to check the raw scores since the CSV only has reasoning/score
features = pd.read_parquet("candidate_features.parquet")
final_100 = pd.merge(df, features, on="candidate_id")

print("--- COVERAGE DISTRIBUTION ---")
print(final_100["coverage_score"].value_counts().sort_index())
print(f"\nCoverage >= 2: {(final_100['coverage_score'] >= 2).sum()}")

elite_count = len(final_100[
    (final_100["retrieval_score"] > 0) |
    (final_100["evaluation_score"] > 0) |
    (final_100["vector_db_score"] > 0)
])
print(f"\n--- ELITE COUNT (Non-Zero Core Symbolic) ---")
print(f"Total: {elite_count} / 100")

df = pd.read_csv("team_redrob_submission.csv")

print("Unique IDs:", df["candidate_id"].nunique())
print("Rows:", len(df))
print("Unique ranks:", df["rank"].nunique())
print("Min rank:", df["rank"].min())
print("Max rank:", df["rank"].max())
print("Duplicate scores:", df["score"].duplicated().sum())