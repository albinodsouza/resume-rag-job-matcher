
# Resume RAG and Job Matching System

## Overview

This project implements a resume retrieval and job-matching pipeline using sentence embeddings, ChromaDB vector storage, BM25 keyword retrieval, and heuristic candidate scoring.

## Features

- Loads resume text files and extracts candidate metadata.
- Splits resumes into smaller section-aware chunks.
- Generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- Stores searchable chunks and metadata in ChromaDB.
- Combines semantic similarity and BM25 keyword scores.
- Applies required-skill and minimum-experience filters.
- Returns ranked candidates with match scores, skills, excerpts, and explanations.
- Evaluates retrieval using Precision@10, Recall@10, and latency.

## Project Structure

```text
resume-rag-project/
├── resume_rag.py
├── job_matcher.py
├── Resume_RAG_Project.ipynb
├── README.md
├── resumes/
│   └── resume_*.txt
├── job_descriptions/
│   └── JD*.json
├── vector_db/
└── evaluation/
    ├── evaluation_metrics.csv
    ├── evaluation_summary.json
    └── sample_matches.json
```

## Requirements

Python 3.10 or later is recommended.

Install dependencies:

```bash
pip install sentence-transformers chromadb rank-bm25 pypdf python-docx pandas scikit-learn
```

## Running the Project

Run commands from the project directory.

Build the resume index:

```bash
python resume_rag.py
```

Run matching against the first job description:

```bash
python job_matcher.py
```

The example matching output is written to `evaluation/sample_matches.json`.

For experimentation and evaluation, open `Resume_RAG_Project.ipynb` in Google Colab or Jupyter and run the notebook cells in order.

## Evaluation

The evaluation script in the notebook reports:

- Precision@10
- Recall@10
- Measured matching latency

Results are saved under `evaluation/`.

## Dataset and Limitations

The supplied resumes are synthetic profiles created for testing. The evaluation uses rule-based relevance labels and does not establish real-world hiring accuracy.

The match score is a heuristic ranking score, not a probability of job success. The current implementation primarily reads TXT resumes; PDF and DOCX libraries are installed but full ingestion support should be added separately.

## Future Improvements

- Add PDF and DOCX resume ingestion.
- Improve skill normalization and synonym matching.
- Compare semantic-only, BM25-only, and hybrid retrieval.
- Tune scoring weights using a manually reviewed dataset.
- Add more resumes and job descriptions.
