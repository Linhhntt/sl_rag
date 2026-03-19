# DATASET: TREC 2021 Health Misinformation

## Overview

This dataset contains health-related documents with ground-truth credibility annotations from the TREC 2021 Health Track misinformation evaluation. The data processing pipeline is documented following `ir_datasets.ipynb`.

## Data Pipeline (ir_datasets.ipynb)

### 1. Extract and Load TREC Evaluation Resources
- Extracts `misinfo-resources-2021.tar.gz` containing TREC health evaluation resources
- Loads QRELS (quality relevance judgments) with credibility assessments
- QRELS file format: query ID, iteration, document ID, relevance, correctness, credibility

### 2. Analyze QRELS Data
- Computes summary statistics across 35 queries
- Average: ~13 judgments per query
- Credibility scores range from 0-2 (where 2 = PubMed-level credibility)
- Documents labeled for:
  - **Relevance** (`rel`): usefulness to the query
  - **Correctness** (`correct`): 0=wrong, 1=partial, 2=correct answer
  - **Credibility** (`cred`): 0=low, 1=medium, 2=high (PubMed-level)

### 3. Data Filtering
Filters QRELS to retain only valid documents:
- Relevance score ≥ 1 (useful documents)
- Correctness and credibility scores in valid range (0-2)
- Excludes error flags and irrelevant judgments

### 4. Load Query Topics
- Parses XML file containing 35 health-related queries
- Each query contributes to the evaluation set

### 5. Extract Relevant Documents from C4
- Groups filtered documents by C4 shard identifier
- Downloads only necessary shards from HuggingFace's C4 collection
- Extracts and saves document metadata to `extracted_docs.jsonl`:
  - `docid`: unique document identifier
  - `text`: document content
  - `url`: source URL
  - `timestamp`: publication timestamp
  - `cred`: credibility score from TREC assessors
- Deletes shard files after extraction to minimize storage

### 6. Patch Missing Data
- Handles gaps in shard coverage by processing shards before/after initial range
- Ensures all relevant documents are included in final output
- Merges prepended documents with existing data

## Output Files

- **`extracted_docs.jsonl`**: Main dataset file containing document text and credibility annotations (6,369 documents)
- **`trec_topics.csv`**: Health-related queries for evaluation: 51 lines (35 health-related queries + 1 header)
- **`trec_health_eval/`**: Directory containing:
  - `qrels/`: Quality relevance judgments
  - `topics/`: Query definitions in XML format
  - `scripts/`: Evaluation and data generation scripts

