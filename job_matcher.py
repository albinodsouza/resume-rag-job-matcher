
from pathlib import Path
import json
import re
import time

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from resume_rag import get_resume_collection, MODEL_NAME


PROJECT_DIR = Path(__file__).resolve().parent
JD_DIR = PROJECT_DIR / "job_descriptions"


def tokenize(text):
    """Tokenize text for BM25 keyword retrieval."""
    return re.findall(r"[a-zA-Z0-9+#.]+", text.lower())


def load_job_description(jd_path):
    with open(jd_path, "r", encoding="utf-8") as file:
        return json.load(file)


def match_job(job_description, top_k=10):
    """
    Rank resumes using semantic retrieval, BM25, skills overlap,
    and experience. Enforce required skills and minimum experience.
    """
    start_time = time.perf_counter()

    if isinstance(job_description, (str, Path)):
        job = load_job_description(job_description)
    else:
        job = job_description

    description = job["description"]
    required_skills = [s.lower() for s in job.get("must_have", [])]
    minimum_experience = int(job.get("minimum_experience", 0))

    collection = get_resume_collection()
    count = collection.count()

    if count == 0:
        raise ValueError("Resume index is empty. Build the index first.")

    # Retrieve all chunks so candidates can be ranked and filtered
    # after combining semantic and keyword evidence.
    stored = collection.get(include=["documents", "metadatas"])

    documents = stored["documents"]
    metadatas = stored["metadatas"]
    ids = stored["ids"]

    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(
        description,
        normalize_embeddings=True
    ).tolist()

    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=count,
        include=["documents", "metadatas", "distances"]
    )

    semantic_distance = {}
    for doc_id, metadata, distance in zip(
        semantic_results["ids"][0],
        semantic_results["metadatas"][0],
        semantic_results["distances"][0]
    ):
        # Chroma's cosine distance is converted to a 0-100 similarity.
        similarity = max(0.0, min(100.0, (1.0 - float(distance)) * 100))
        path = metadata["resume_path"]
        semantic_distance[path] = max(
            semantic_distance.get(path, 0.0), similarity
        )

    # BM25 scores are calculated over indexed resume chunks.
    tokenized_documents = [tokenize(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)
    bm25_scores = bm25.get_scores(tokenize(description))

    bm25_by_resume = {}
    for doc_id, metadata, score in zip(ids, metadatas, bm25_scores):
        path = metadata["resume_path"]
        bm25_by_resume[path] = max(
            bm25_by_resume.get(path, 0.0), float(score)
        )

    max_bm25 = max(bm25_by_resume.values(), default=0.0)

    # Group the indexed chunks by candidate resume.
    candidates = {}

    for metadata in metadatas:
        path = metadata["resume_path"]
        if path not in candidates:
            candidates[path] = {
                "candidate_name": metadata["candidate_name"],
                "experience_years": int(metadata["experience_years"]),
                "education": metadata["education"],
                "skills": json.loads(metadata["skills"]),
                "resume_path": path,
                "chunks": []
            }

    for doc, metadata in zip(documents, metadatas):
        candidates[metadata["resume_path"]]["chunks"].append(doc)

    ranked = []

    for path, candidate in candidates.items():
        candidate_skills = [
            skill.lower() for skill in candidate["skills"]
        ]

        matched_skills = [
            skill for skill in candidate["skills"]
            if skill.lower() in description.lower()
        ]

        # A skill can be required even when the job description's
        # wording varies, so check the must-have list separately.
        missing_required = [
            skill for skill in required_skills
            if skill not in candidate_skills
        ]

        eligible = (
            not missing_required
            and candidate["experience_years"] >= minimum_experience
        )

        if not eligible:
            continue

        semantic_score = semantic_distance.get(path, 0.0)
        lexical_score = (
            bm25_by_resume.get(path, 0.0) / max_bm25 * 100
            if max_bm25 > 0 else 0.0
        )

        # Skills overlap is measured against job-mentioned skills.
        job_skill_terms = set(tokenize(description))
        candidate_skill_terms = set(tokenize(" ".join(candidate["skills"])))
        relevant_terms = job_skill_terms.intersection(candidate_skill_terms)
        skills_score = (
            len(relevant_terms) / max(len(candidate_skill_terms), 1) * 100
        )

        experience_score = min(
            100.0,
            candidate["experience_years"] /
            max(minimum_experience, 1) * 100
        )

        # Heuristic score, not an objective measure of hiring suitability.
        match_score = (
            0.40 * semantic_score
            + 0.25 * lexical_score
            + 0.25 * skills_score
            + 0.10 * experience_score
        )

        excerpts = [
            chunk for chunk in candidate["chunks"]
            if any(skill.lower() in chunk.lower()
                   for skill in matched_skills)
        ]

        if not excerpts:
            excerpts = candidate["chunks"][:2]

        reasoning = (
            f"Meets the minimum experience requirement of "
            f"{minimum_experience} years and has all required skills. "
            f"Semantic score: {semantic_score:.1f}/100; "
            f"keyword score: {lexical_score:.1f}/100."
        )

        ranked.append({
            "candidate_name": candidate["candidate_name"],
            "resume_path": candidate["resume_path"],
            "match_score": round(match_score, 2),
            "matched_skills": matched_skills,
            "relevant_excerpts": excerpts[:3],
            "reasoning": reasoning,
            "experience_years": candidate["experience_years"],
            "education": candidate["education"],
            "semantic_score": round(semantic_score, 2),
            "keyword_score": round(lexical_score, 2),
            "eligible": True
        })

    ranked.sort(key=lambda item: item["match_score"], reverse=True)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "job_description": {
            "job_id": job.get("job_id", "CUSTOM"),
            "title": job.get("title", "Untitled job"),
            "description": description,
            "must_have": job.get("must_have", []),
            "minimum_experience": minimum_experience
        },
        "total_candidates_indexed": len(candidates),
        "eligible_candidates": len(ranked),
        "top_matches": ranked[:top_k],
        "latency_ms": elapsed_ms,
        "scoring_note": (
            "Scores are heuristic ranking estimates, not probabilities "
            "of job success or objective hiring suitability."
        )
    }


def match_job_file(job_file, top_k=10, output_file=None):
    result = match_job(job_file, top_k=top_k)

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8"
        )

    return result


if __name__ == "__main__":
    job_files = sorted(JD_DIR.glob("*.json"))

    if not job_files:
        raise FileNotFoundError("No job description JSON files found.")

    result = match_job_file(
        job_files[0],
        top_k=10,
        output_file=PROJECT_DIR / "evaluation" / "sample_matches.json"
    )

    print(json.dumps(result, indent=2))
