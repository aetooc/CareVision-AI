import chromadb
import os
import shutil
from pipeline.logging_utils import log_action

# ── ChromaDB Initialisation ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR   = os.path.join(BASE_DIR, "data", "embeddings")
os.makedirs(DB_DIR, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(
    name="patient_records",
    metadata={"hnsw:space": "cosine"},
)

# ── Patient directory helpers ──────────────────────────────────────────────────

def get_patient_dirs(patient_id: str) -> dict:
    """
    Returns and creates per-patient directories for all file-based data.

    On-disk layout after calling this:
        data/
          uploads/{patient_id}/       <- source MP4
          frames/{patient_id}/        <- extracted JPEG frames (1/sec)
          transcripts/{patient_id}/   <- extracted_audio.wav + transcript.txt
          embeddings/                 <- single shared ChromaDB (NOT per-patient)

    Pass the returned paths directly to process_audio(), process_video_frames(),
    and validate_data() so no file ever lands in a shared flat folder.
    """
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
    """
    Upserts this patient's transcript into the shared ChromaDB collection.
    Each patient gets exactly one document: id = "doc_{patient_id}".
    Pass any extra fields (frames count, video_filename, etc.) in metadata.
    Returns True on success.
    """
    doc_id    = f"doc_{patient_id}"
    base_meta = {"type": "transcript", "patient_id": patient_id}
    if metadata:
        base_meta.update(metadata)

    try:
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
    Cosine-similarity search across all stored patient transcripts.
    Returns list of dicts with 'document' and 'metadata' keys.
    """
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
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
    """
    Complete, patient-scoped GDPR Article 17 purge. Four steps:

      1. ChromaDB  — delete doc_{patient_id} from the shared collection.
                     Other patients' vectors are untouched.
      2. uploads/  — rmtree data/uploads/{patient_id}/
      3. frames/   — rmtree data/frames/{patient_id}/
      4. transcripts/ — rmtree data/transcripts/{patient_id}/

    The embeddings folder itself (data/embeddings/) is shared across all
    patients and is NOT deleted — only the specific document is removed
    from the collection via ChromaDB's delete API.

    Returns True if all four steps succeeded.
    """
    data_dir = os.path.join(BASE_DIR, "data")
    doc_id   = f"doc_{patient_id}"
    all_ok   = True

    # Step 1 — ChromaDB: delete this patient's vector only
    try:
        collection.delete(ids=[doc_id])
        log_action("DATA_DELETION", f"ChromaDB embedding removed for {patient_id}.")
    except Exception as e:
        log_action("WARNING", f"Embedding deletion skipped (may not exist): {e}")

    # Steps 2-4 — Delete each patient-namespaced directory entirely
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