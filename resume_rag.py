
from pathlib import Path
import re
import json
import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path("/content/resume-rag-project")
RESUME_DIR = PROJECT_DIR / "resumes"
VECTOR_DB_DIR = PROJECT_DIR / "vector_db"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def extract_metadata(text, resume_path):
    """Extract basic candidate metadata from resume text."""
    name_match = re.search(r"(?im)^Name:\s*(.+)$", text)
    exp_match = re.search(r"(?im)^Experience:\s*(\d+)\s*years?", text)
    education_match = re.search(r"(?im)^Education:\s*(.+)$", text)
    skills_match = re.search(r"(?im)^Skills:\s*(.+)$", text)

    return {
        "candidate_name": name_match.group(1).strip() if name_match else "Unknown",
        "experience_years": int(exp_match.group(1)) if exp_match else 0,
        "education": education_match.group(1).strip() if education_match else "Not specified",
        "skills": [
            skill.strip() for skill in skills_match.group(1).split(",")
        ] if skills_match else [],
        "resume_path": str(resume_path),
    }


def split_into_chunks(text, max_chars=900):
    """Split a resume by sections, then split long sections into smaller chunks."""
    sections = re.split(
        r"(?im)(?=^(?:Name|Target Role|Experience|Education|Skills|Professional Summary|Experience Details):)",
        text
    )

    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # Split long sections at paragraph or sentence boundaries.
            paragraphs = re.split(r"\n+|(?<=[.!?])\s+", section)
            current = ""

            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue

                if len(current) + len(paragraph) + 1 > max_chars and current:
                    chunks.append(current.strip())
                    current = paragraph
                else:
                    current = f"{current}\n{paragraph}".strip()

            if current:
                chunks.append(current.strip())

    return chunks


def build_resume_index(reset=True):
    """Read resume files, embed chunks and save them in ChromaDB."""
    if not RESUME_DIR.exists():
        raise FileNotFoundError(f"Resume directory not found: {RESUME_DIR}")

    resume_paths = sorted(RESUME_DIR.glob("*.txt"))

    if not resume_paths:
        raise ValueError("No .txt resumes found in the resumes folder.")

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))

    if reset:
        try:
            client.delete_collection("resume_chunks")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name="resume_chunks",
        metadata={"hnsw:space": "cosine"}
    )

    model = SentenceTransformer(MODEL_NAME)

    ids, documents, metadatas = [], [], []
    total_chunks = 0

    for resume_path in resume_paths:
        text = resume_path.read_text(encoding="utf-8")
        candidate = extract_metadata(text, resume_path)
        chunks = split_into_chunks(text)

        for chunk_number, chunk in enumerate(chunks):
            chunk_id = f"{resume_path.stem}_chunk_{chunk_number}"

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "candidate_name": candidate["candidate_name"],
                "experience_years": candidate["experience_years"],
                "education": candidate["education"],
                "skills": json.dumps(candidate["skills"]),
                "resume_path": str(resume_path),
                "chunk_number": chunk_number,
            })

        total_chunks += len(chunks)

    # Batch embeddings to avoid processing the entire dataset at once.
    batch_size = 64

    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start:start + batch_size]
        batch_ids = ids[start:start + batch_size]
        batch_metadata = metadatas[start:start + batch_size]

        embeddings = model.encode(
            batch_docs,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metadata,
            embeddings=embeddings
        )

    return {
        "resume_count": len(resume_paths),
        "chunk_count": total_chunks,
        "indexed_chunk_count": collection.count(),
        "collection": collection.name,
        "database_path": str(VECTOR_DB_DIR),
    }


def get_resume_collection():
    """Open the already-built resume collection."""
    client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
    return client.get_collection("resume_chunks")


if __name__ == "__main__":
    result = build_resume_index()
    print(json.dumps(result, indent=2))
