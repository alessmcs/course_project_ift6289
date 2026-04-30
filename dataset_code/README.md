# Dataset Construction Pipeline

This repository implements a pipeline to build a dataset of aligned **code–documentation pairs** and generate **question–answer samples**.

You may use our Zenodo repository to access the jsonl files to which our scripts write, here: https://zenodo.org/records/19828242?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjYzMTMyZjExLTAxMTUtNGExNS04NDlhLWFhM2NiODI5ZWM0YSIsImRhdGEiOnt9LCJyYW5kb20iOiI5NGE5Y2UzNmU4ODhlY2FmOTVlYzJhY2QxYTAwODNjYyJ9.C9sYXLi9b4ykAz7msKZwDcogdLIbxTbfX0icBoS5FOQb5TEtRPlQcp9FoBxymF5PzYj9ByowN50NIcVQEbP8Ng 
---

## Pipeline Overview

The pipeline consists of the following steps:

1. Repository discovery  
2. Repository filtering and file extraction  
3. Documentation entity extraction and alignment  
4. Dataset sampling  
5. Alignment verification (LLM-as-judge)  
6. Question generation  
7. Train/validation/test split  

---

## Steps

### 1. Repository Discovery

**`repo_discovery.py`**
- Saves ~292 candidate repositories  

---

### 2. Repository Filtering & File Extraction

**`get_repo_files.py`**
- Reads candidate repositories  
- Filters based on:
  - License  
  - API density  
- Outputs:
  - `accepted_repos.jsonl`
  - `accepted_repo_files.jsonl` (file contents)

---

### 3. Diagnostics (Optional)

**`diagnostic_repo_stats.py`**
- Provides a quick overview of accepted/rejected repositories  
- Avoids running the full pipeline  

---

### 4. Repository Quality Evaluation (Unused)

**`eval_repo_quality.py`**
- Scores repositories based on:
  - Core file density  
  - Aggregated file metrics  
  - Commit details and their impact on core files  

**`file_metrics.py`**
- Computes metrics at the file level  
- Extracts documentation entities  
- Outputs:
  - `file_entities.jsonl`

**Note:**
- Output file `repo_scores.jsonl` is not used  
- Scoring logic remains commented out due to formatting issues  

---

### 5. Documentation Alignment

**`alignment.py`**
- Processes documentation entities  
- Performs alignment based on:
  - Documentation type  
  - Confidence levels (strong / medium)  
- Outputs:
  - `final_aligned_dataset.jsonl`

---

### 6. External Data (CodeSearchNet)

**`csn_samples.py`**
- Retrieves ~2k pairs from CodeSearchNet:
  - Python (docstrings)  
  - Java (Javadoc)  
- Outputs:
  - `csn_samples.jsonl`

---

### 7. Training Sample Selection

**`training_samples.py`**
- Subsamples entities based on predefined quotas  
- Reduces dataset size for efficient LLM usage  
- Outputs:
  - `training_dataset_final.jsonl`

---

### 8. Alignment Verification (LLM-as-Judge)

**`llm_judge_pilot.py`**
- Validates alignment quality using an LLM  
- Input:
  - `training_dataset_final.jsonl` (~8100 samples)  
- Filters low-quality pairs  
- Outputs:
  - `training_dataset_verified.jsonl` (~6400 samples)  
  - `pilot_judgements.jsonl`

---

### 9. Question Generation

**`gen_questions/inference.py`**
- Generates questions for the verified dataset  
- Outputs:
  - `results.jsonl`

---

### 10. Dataset Splitting

**`split_dataset.py`**
- Splits dataset into:
  - Train / Validation / Test  
- Uses **repo-wise splitting** to avoid data leakage  
- Input:
  - `results.jsonl`  
- Outputs:
  - train / val / test datasets  

---

## Final Outputs

- `final_aligned_dataset.jsonl`
- `training_dataset_verified.jsonl`
- `results.jsonl`
- Train / validation / test splits  

---

## Notes

- Repository scoring is currently disabled  
- LLM-based filtering ensures dataset quality  
- Dataset combines:
  - Automatically aligned data  
  - CodeSearchNet samples  
