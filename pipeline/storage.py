import chromadb
import os
import shutil
from pipeline.logging_utils import log_action

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR   = os.path.join(BASE_DIR, "data", "embeddings")
os.makedirs(DB_DIR, exist_ok=True)

# ── Lazy ChromaDB init ─────────────────────────────────────────────────────────
# Do NOT initialise at module level — Streamlit Cloud runs imports in a
# restricted environment where touching disk at import time causes errors.
_chroma_client = None
_collection    = None

def _get_collection():
    """Returns the ChromaDB collection, creating it on first call."""
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=DB_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name="patient_records",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Patient directory helpers ──────────────────────────────────────────────────

def get_patient_dirs(patient_id: str) -> dict:
    data_dir = os.path.join(BASE_DIR, "data")
    dirs = {
        "uploads":     os.path.join(data_dir, "uploads",     patient_id),
        "frames":      os.path.join(data_dir, "frames",      patient_id),
        "transcripts": os.path.join(data_dir, "transcripts", patient_id),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


# ── Embeddings ─────────────────────────────────────────────────────────────────

def store_embeddings(patient_id: str, transcript: str, metadata: dict | None = None) -> bool:
    doc_id    = f"doc_{patient_id}"
    base_meta = {"type": "transcript", "patient_id": patient_id}
    if metadata:
        base_meta.update(metadata)
    try:
        _get_collection().upsert(
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
    try:
        results = _get_collection().query(query_texts=[query], n_results=n_results)
        hits = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            hits.append({"document": doc, "metadata": meta})
        log_action("DATA_QUERY", f"Semantic search returned {len(hits)} results.")
        return hits
    except Exception as e:
        log_action("ERROR", f"Semantic search failed: {e}", status="FAILED")
        return []


# ── GDPR Article 17 ────────────────────────────────────────────────────────────

def execute_right_to_be_forgotten(patient_id: str) -> bool:
    data_dir = os.path.join(BASE_DIR, "data")
    doc_id   = f"doc_{patient_id}"
    all_ok   = True

    try:
        _get_collection().delete(ids=[doc_id])
        log_action("DATA_DELETION", f"ChromaDB embedding removed for {patient_id}.")
    except Exception as e:
        log_action("WARNING", f"Embedding deletion skipped (may not exist): {e}")

    for folder in ("uploads", "frames", "transcripts"):
        patient_dir = os.path.join(data_dir, folder, patient_id)
        try:
            if os.path.isdir(patient_dir):
                shutil.rmtree(patient_dir)
                log_action("DATA_DELETION", f"Deleted data/{folder}/{patient_id}/")
            else:
                log_action("WARNING", f"data/{folder}/{patient_id}/ not found — already deleted?")
        except Exception as e:
            log_action("ERROR", f"Failed to delete data/{folder}/{patient_id}/: {e}", status="FAILED")
            all_ok = False

    status = "COMPLIANCE_SUCCESS" if all_ok else "COMPLIANCE_PARTIAL"
    log_action(status, f"Right-to-be-Forgotten complete for patient {patient_id}.")
    return all_ok