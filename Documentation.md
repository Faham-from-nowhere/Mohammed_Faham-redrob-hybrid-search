# Redrob Hybrid Search XAI Ranker: Technical Documentation


## 1. Executive Summary & System Architecture

Our solution is a **Dual-Engine Hybrid Ranking System** that evaluates 100,000 candidates against a Senior AI Engineer job description. It is built to strictly adhere to the challenge's constraints: 5-minute maximum CPU runtime, zero external API calls during the ranking step, and extreme robustness against "honeypots" (keyword-stuffed or impossible profiles).

The architecture fuses two distinct pipelines:
1. **The Symbolic Engine (Sparse):** Extracts explicit evidence of skills, utilizing IDF-weighted exact matches and verb-based heuristics (builder vs. supporter) from the candidate's career history and summary.
2. **The Semantic Engine (Dense):** Projects both the candidate descriptions and independent Job Description (JD) axes into a shared dense vector space using `SentenceTransformers` (`all-MiniLM-L6-v2`) and matches them via `FAISS`.

The final ranking is determined through **Reciprocal Rank Fusion (RRF)**, aggressively penalized by a behavioral and experience-based **Trust Score**.

### Core Tenets
- **Explainable AI (XAI):** Every candidate's score is deterministic and can be broken down into exact terms, semantic distances, and penalization flags.
- **Product Over Pure Research:** We optimize for what recruiters actually need—genuine builders with verified experience, penalizing academic-only profiles or candidates demonstrating "LLM tutorial" buzzwords without production depth.
- **Semantic Fratricide Avoidance:** By applying a strict Jaccard-diversity filter during final selection, the system ensures a varied cohort, preventing the top 100 from being filled with clones sharing identical buzzwords.

---

## 2. Full Application Flow (A to Z)

The system operates in a multi-stage pipeline designed for extreme efficiency.

1. **Pre-Processing & IDF Weighting (`build_rarity_weights.py`)**
   - **Action:** Scans the 100K `candidates.jsonl` to build Inverse Document Frequency (IDF) weights for key target terms (e.g., `ndcg`, `milvus`, `rag`).
   - **Purpose:** Rewards candidates possessing rare but highly relevant skills, ensuring commodity terms don't overpower niche expertise.

2. **Semantic Embedding & Indexing (`build_semantic.py`)**
   - **Action:** Embeds the candidate's full written text (Summary + Career) using an offline CPU-bound `SentenceTransformer`. The JD is split into 5 distinct "Axes" (Retrieval, Vector DB, Evaluation, Production, Startup DNA).
   - **Purpose:** Calculates cosine similarity scores for each candidate against each of the 5 JD axes and identifies "Semantic Coverage" (how many axes a candidate is in the top 5% for). Outputs `semantic_features.parquet` and a `faiss.index`.

3. **Symbolic Extraction & Trust Scoring (`build_features.py`)**
   - **Action:** Uses `FlashText` to perform lightning-fast exact keyword extraction against predefined clusters. 
   - **Action:** Calculates a critical **Trust Score**. This score penalizes suspicious attributes: $>30$ years of experience, extremely long notice periods, unrelated job titles (e.g., "HR Manager" claiming RAG experience), or suspicious keywords ("asked chatgpt"). 
   - **Purpose:** Generates raw cluster scores weighted by JD importance and term rarity. Outputs `candidate_features.parquet`.

4. **Fusion, Diversity Filtering, and Ranking (`rank.py`)**
   - **Action:** Merges the symbolic and semantic parquets. 
   - **Action:** Applies a hard quarantine (dropping low trust candidates).
   - **Action:** Applies **Reciprocal Rank Fusion (RRF)**, combining the independent rank positions of a candidate across retrieval, evaluation, vector DBs, production ML, and their semantic core.
   - **Action:** Enforces **Diversity Filter Version C** (a Jaccard similarity threshold) to drop candidates who are near-identical textual clones of already selected candidates.
   - **Output:** The final `team_redrob_submission.csv` containing the top 100 candidates with dynamic reasoning strings.

5. **Validation & Telemetry (`validate_submission.py`, `app.py`)**
   - **Action:** The submission is validated against the hackathon rules.
   - **Action:** Telemetry and results can be inspected via the Streamlit XAI Dashboard (`app.py`), enabling deep dives into *why* a candidate was ranked where they were.

---

## 3. File-by-File Breakdown & Purpose

### Core Pipeline
- **`build_features.py`**
  - **What it does:** Extracts sparse keyword features and calculates the Trust Score, Coverage Score, Market Score (search appearances + saves), and Builder Ratio.
  - **Why:** To map explicit textual claims to hard scores. It protects the pipeline from keyword stuffers by checking if listed skills actually appear in the candidate's descriptive text.

- **`build_semantic.py`**
  - **What it does:** Runs the `all-MiniLM-L6-v2` transformer model to create dense embeddings for candidate text and computes their similarity against 5 discrete vectors representing the core JD requirements.
  - **Why:** Because a Tier 5 candidate might build a "hybrid candidate matching pipeline" without ever using the exact string "semantic search." Dense embeddings catch conceptual matches.

- **`build_rarity_weights.py`**
  - **What it does:** A pre-computation script that dynamically generates `term_rarity_weights.json` based on document frequency across the entire 100K corpus.
  - **Why:** Hardcoding term weights is brittle. A dynamic IDF ensures that as industry vocabulary shifts, the system naturally rewards rarer, highly specific terms.

- **`rank.py`**
  - **What it does:** The final arbiter. Fuses the features, drops the honeypots, runs the diversity algorithm, applies the core domain guardrails, and generates the final CSV submission.
  - **Why:** Centralizes the business logic of ranking. It guarantees that the submitted candidates actually match the specific constraints of the hackathon (e.g., must be a "builder," must not be a keyword-stuffer).

### UI, Analytics & Auditing
- **`app.py` (The XAI Dashboard)**
  - **What it does:** A Streamlit dashboard offering Candidate Inspection, Head-to-Head Comparisons, and System Telemetry.
  - **Why:** A black-box AI is useless to a recruiter. The UI proves to stakeholders exactly how the RRF algorithm weighed symbolic vs. semantic signals to arrive at a decision.

- **`eda.py` & `eda2.py`**
  - **What it does:** Exploratory Data Analysis scripts that parse the 100K dataset to generate statistics (`field_stats.json`, `semantic_stats.json`) and flag structural anomalies.
  - **Why:** Used to discover the schema limits (e.g., maximum experience years, notice period distributions) to accurately calibrate the Trust Score penalties in `build_features.py`.

- **`feature_extractor.py`**
  - **What it does:** Runs an audit matrix comparing "Claimed Skills" vs. "Supported Evidence" (did they list FAISS in their skills array, but never talk about it in their career history?).
  - **Why:** Essential for identifying "Humble Experts" (didn't list the skill, but did the work) and "Stuffers" (listed the skill, never did the work).

- **`semantic_check.py`, `check_submission.py`, `audit_parquet.py`, `audit_parquet2.py`**
  - **What they do:** A suite of CLI diagnostic tools to peek into the generated parquet files, verify semantic distributions, check RRF overlap matrices, and validate candidate outputs.
  - **Why:** Rapid iteration. We needed to constantly verify that our normalization algorithms weren't flattening out the scores.

### Validation & Context
- **`validate_submission.py`**
  - **What it does:** The official challenge validator script to ensure the final CSV adheres strictly to the 100-row, monotonically decreasing score requirements.
- **Context Docs (`README.txt`, `job_description.txt`, `redrob_signals_doc.txt`, `submission_spec.txt`)**
  - **What they are:** Hackathon specs detailing the target persona, the behavioral traps (honeypots), and the compute limits.

---

## 4. Key Architectural Decisions (The "Why")

> [!CAUTION]
> **Why not just use an LLM (like GPT-4) per candidate?**
> The spec explicitly forbids external API calls during ranking, and local LLM inference on 100K candidates would catastrophically violate the 5-minute CPU constraint. We opted for a pre-computed feature extraction and FAISS approach, which ranks 100K candidates in mere seconds.

### Reciprocal Rank Fusion (RRF) over Weighted Averages
We avoid averaging raw scores (e.g., $0.6$ Semantic $+ 0.4$ Symbolic) because dense cosine similarities and sparse TF-IDF scores live in vastly different mathematical distributions. Normalizing them perfectly is impossible. RRF converts raw scores to rank positions ($1/K + rank$) before fusing, creating an incredibly stable ensemble that prevents one extreme signal from dominating.

### The "Builder Ratio" via FlashText
We mapped verbs into two clusters: "builders" (architected, built, shipped) and "supporters" (assisted, explored). Instead of heavy NLP dependency parsing, we used `FlashText` (which is $10\times$ faster than regex) to calculate a "Builder Ratio". This perfectly aligns with the JD's request for a "scrappy product-engineering attitude" over a "research-only" profile.

### Diversity Filtering (Preventing Semantic Fratricide)
In standard retrieval systems, if 10 engineers from the same company apply with the exact same resume template, they will occupy ranks 1 through 10. We implemented an A/B/C tested Jaccard Similarity threshold ($0.99$ overlap with minimum 3 term checks) in `rank.py`. This ensures we submit a highly diverse top 100 pool, maximizing the surface area of potential hires.

---

## 5. Evaluation & Safeguards against Traps

The dataset explicitly included traps: Keyword stuffers, non-engineers with AI keywords, and "Honeypots" (impossible profiles).

**Our Defensive Layers:**
1. **The Trust Score:** 
   - Deducts points for impossible metrics ($>30$ years experience, $>180$ day notice period).
   - Deducts points if the current title is completely disjointed from engineering (e.g., "Marketing Manager").
   - Analyzes the `Claimed vs. Supported` ratio. If a candidate lists 20 AI skills but their career history contains 0 mentions of them, their Trust Score plummets.
2. **The Minimum Trust Guardrail:** Candidates with a Trust Score $< 0.6$ are dropped entirely in `rank.py` before RRF fusion even begins.
3. **Core Domain Requirement:** To prevent generic software engineers from slipping through on purely behavioral metrics, `rank.py` enforces that every final candidate MUST have a non-zero sparse score in Retrieval, Vector DBs, or Evaluation.

---

## 6. App & Telemetry (The XAI Dashboard)

To ensure this tool is usable by recruiting stakeholders, we built `app.py`. The dashboard visualizes the "Why" behind the ranking.

- **Candidate Inspection:** Shows a candidate's Final Rank, RRF Score, and a bar chart breaking down exactly how they performed across the 5 JD axes.
- **Verified Evidence Visualization:** Displays a Plotly chart of the exact sparse terms found in their resume, providing instant proof of their background.
- **Head-to-Head Matchup:** Allows recruiters to put Candidate A against Candidate B to see precisely where one outscored the other.
- **System Telemetry:** Tracks global metrics like "Mean Pool Trust" and "Core Domain Elites" to prove the health and validity of the final Top 100 pool.

---


## 1. Why RRF instead of a weighted sum?

 We chose Reciprocal Rank Fusion because our signals exist on very different scales. For example, evaluation scores can reach 70+, while production scores may only reach 20. A weighted sum would require extensive normalization and tuning, and a single dominant metric could overwhelm the ranking.

 RRF operates on ranks instead of raw scores. It rewards candidates who consistently rank well across multiple dimensions rather than excelling in only one area.
 In our use case, we wanted to identify "triple-threat" candidates with strong retrieval, evaluation, and vector database expertise. RRF naturally favors these candidates.
 A weighted sum answers "Who has the highest score?" while RRF answers "Who is consistently strong everywhere?" For hiring, consistency across critical skills was more important.

---

## 2. Why FAISS instead of pure BM25?

 BM25 is excellent for exact keyword matching but struggles with semantic equivalence.
 For example, a candidate may describe building a "dense retrieval system" without mentioning FAISS explicitly. BM25 may miss that relationship entirely.

 FAISS allows us to perform dense vector similarity search on candidate career histories and identify candidates whose experience is semantically aligned with the JD even when exact keywords are absent.

 We therefore combined both approaches:
 * Sparse symbolic matching verifies explicit evidence.
 * Dense semantic retrieval discovers contextual expertise.

 The final ranking fuses both perspectives.


---

## 3. How is trust score computed?

 The trust score is based on internal consistency checks within the profile.
 We compare claimed skills against career duration and evidence density. Profiles with unrealistic skill-to-experience ratios receive penalties.
 For example, a profile claiming expertise across many advanced domains while having minimal experience history receives a lower trust score.

 We use trust in two ways:
 1. Profiles below 0.6 are filtered out completely.
 2. Remaining profiles receive a trust multiplier during rank fusion.


---

## 4. Why a 0.99 Jaccard threshold?

 We initially tested stricter diversity thresholds such as 0.95.
 However, diagnostics showed that many elite candidates were being removed simply because they shared common industry terminology such as FAISS, Pinecone, RAG, and Kubernetes.

 We ran an A/B/C diversity experiment:
 * Strict filtering reduced elite candidates from 91 to 41.
 * No filtering preserved all elites but increased redundancy.
 * A relaxed 0.99 threshold retained 91 elite candidates while still reducing exact duplicates.

 Therefore, 0.99 was selected empirically based on observed ranking quality.

---

## 5. What happens when a candidate has semantic relevance but zero keyword matches?


 Those candidates are still discoverable through the semantic engine.
 The semantic pipeline embeds candidate career histories using all-MiniLM-L6-v2 and compares them against five JD-specific semantic axes.
 If the candidate demonstrates strong contextual alignment, they can achieve a high semantic rank even with zero symbolic matches.
 However, because we fuse semantic and symbolic rankings, a candidate generally performs best when they exhibit both contextual expertise and explicit evidence.
 This prevents pure keyword dependence while also preventing semantic false positives.

---

## 6. How would this scale to 10 million resumes?


 The architecture was intentionally designed around pre-computed artifacts.
 Feature extraction and embedding generation are performed offline and stored in Parquet format.

 During ranking:
 * We operate only on pre-computed features.
 * FAISS indexes support highly efficient nearest-neighbor retrieval.
 * RRF is computationally lightweight.

 For larger deployments we would:
 * Replace Flat FAISS with IVF or HNSW indexes.
 * Partition candidate embeddings.
 * Run embedding generation as a distributed batch process.
 * Store artifacts in a data lake rather than local files.

 The online ranking stage would remain fast because the expensive embedding computation occurs offline.




 We initially assumed stricter diversity filtering would improve candidate quality. Our experiments showed the opposite. A 0.95 Jaccard threshold removed many top candidates because elite Search/ML engineers naturally share similar vocabulary. The data taught us that preserving high-quality candidates was more important than maximizing vocabulary diversity.
