import chromadb
import os
from pipeline.logging_utils import log_action

# ── ChromaDB Initialisation ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "data", "embeddings")
os.makedirs(DB_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(
    name="patient_records",
    metadata={"hnsw:space": "cosine"},   # cosine similarity for text embeddings
)


def store_embeddings(patient_id: str, transcript: str, metadata: dict | None = None) -> bool:
    """
    Upserts the transcript (and optional extra metadata) into ChromaDB.
    ChromaDB uses its default sentence-transformer embedding function.
    Returns True on success.
    """
    doc_id = f"doc_{patient_id}"
    base_meta = {"type": "transcript", "patient_id": patient_id}
    if metadata:
        base_meta.update(metadata)

    try:
        # upsert handles re-processing without duplicate key errors
        collection.upsert(
            documents=[transcript],
            metadatas=[base_meta],
            ids=[doc_id],
        )
        log_action("DATA_STORAGE", f"Transcript upserted in ChromaDB (id={doc_id}).")
        return True
    except Exception as e:
        log_action("ERROR", f"Embedding storage failed: {e}", status="FAILED")
        return False


def semantic_search(query: str, n_results: int = 3) -> list[dict]:
    """
    Runs a semantic similarity search over stored transcripts.
    Returns a list of result dicts with 'document' and 'metadata' keys.
    """
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        hits = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            hits.append({"document": doc, "metadata": meta})
        log_action("DATA_QUERY", f"Semantic search returned {len(hits)} results for query.")
        return hits
    except Exception as e:
        log_action("ERROR", f"Semantic search failed: {e}", status="FAILED")
        return []


def execute_right_to_be_forgotten(patient_id: str) -> bool:
    """
    GDPR Article 17 compliance pipeline:
      1. Delete vector embeddings from ChromaDB.
      2. Purge all physical files (video, audio, frames, transcripts).
    Returns True if all steps succeeded.
    """
    DATA_DIR = os.path.join(BASE_DIR, "data")
    all_ok = True

    # Step 1 — Vector deletion ─────────────────────────────────────────────────
    try:
        collection.delete(ids=[f"doc_{patient_id}"])
        log_action("DATA_DELETION", f"Vector embeddings removed for patient {patient_id}.")
    except Exception as e:
        log_action("WARNING", f"Embedding deletion skipped (may not exist): {e}")

    # Step 2 — Physical file purge ─────────────────────────────────────────────
    for folder in ["uploads", "transcripts", "frames"]:
        folder_path = os.path.join(DATA_DIR, folder)
        try:
            for filename in os.listdir(folder_path):
                fp = os.path.join(folder_path, filename)
                if os.path.isfile(fp):
                    os.remove(fp)
            log_action("DATA_DELETION", f"All files purged from /data/{folder}/.")
        except Exception as e:
            log_action("ERROR", f"Failed to clear {folder}: {e}", status="FAILED")
            all_ok = False

    status = "COMPLIANCE_SUCCESS" if all_ok else "COMPLIANCE_PARTIAL"
    log_action(status, f"Right-to-be-Forgotten executed for patient {patient_id}.")
    return all_ok
