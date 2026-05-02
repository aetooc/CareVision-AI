import streamlit as st
import os
import json
import uuid

from pipeline.audio import process_audio
from pipeline.video import process_video_frames
from pipeline.validation import validate_data
from pipeline.agent import generate_clinical_summary, ClinicalChatAgent
from pipeline.storage import get_patient_dirs, store_embeddings, semantic_search, execute_right_to_be_forgotten
from pipeline.logging_utils import log_action
from pipeline.voice import render_voice_chat_ui

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR       = os.path.join(BASE_DIR, "logs")
AUDIT_LOG_FILE = os.path.join(LOGS_DIR, "audit_log.json")

os.makedirs(LOGS_DIR, exist_ok=True)

if not os.path.exists(AUDIT_LOG_FILE):
    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump([], f)

# ── Page config & custom CSS ───────────────────────────────────────────────────
st.set_page_config(
    page_title="CareVision AI · DEEM Lab",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Fonts & palette ──────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --primary: #0d5c6b;
    --accent:  #00c2a8;
    --warn:    #e05c2f;
    --bg:      #f5f7f8;
    --card:    #ffffff;
    --border:  #d6dde2;
    --text:    #1a2730;
    --muted:   #5a7080;
}

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--text); }

/* Header strip */
.cv-header {
    background: linear-gradient(135deg, var(--primary) 0%, #0a3e4a 100%);
    padding: 1.6rem 2rem 1.3rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.cv-header h1 {
    font-family: 'DM Serif Display', serif;
    color: #ffffff;
    font-size: 2rem;
    margin: 0;
    letter-spacing: -0.5px;
}
.cv-header p { color: rgba(255,255,255,0.72); margin: 0; font-size: 0.88rem; font-weight: 300; }
.cv-badge {
    background: rgba(0,194,168,0.18);
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 20px;
    padding: 3px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.73rem;
    white-space: nowrap;
}

/* Cards */
.cv-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    color: inherit;
}
.cv-card h3 { font-family: 'DM Serif Display', serif; font-size: 1.1rem; margin: 0 0 .6rem; color: inherit; }

/* Metric chips */
.metric-row { display: flex; gap: .7rem; flex-wrap: wrap; margin: .8rem 0; }
.metric-chip {
    background: rgba(0,194,168,0.1);
    border: 1px solid rgba(0,194,168,0.25);
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 0.82rem;
    font-family: 'DM Mono', monospace;
    color: inherit;
}
.metric-chip span { color: var(--accent); font-weight: 600; }

/* Audit log entries */
.log-success { border-left: 3px solid var(--accent); padding: .4rem .8rem; margin:.3rem 0; font-size:.8rem; font-family:'DM Mono',monospace; background:rgba(0,194,168,0.08); border-radius:0 6px 6px 0; color:inherit; }
.log-delete  { border-left: 3px solid var(--warn);   padding: .4rem .8rem; margin:.3rem 0; font-size:.8rem; font-family:'DM Mono',monospace; background:rgba(224,92,47,0.08); border-radius:0 6px 6px 0; color:inherit; }
.log-error   { border-left: 3px solid #e05c2f;       padding: .4rem .8rem; margin:.3rem 0; font-size:.8rem; font-family:'DM Mono',monospace; background:rgba(224,92,47,0.08); border-radius:0 6px 6px 0; color:inherit; }

/* Session ID pill */
.session-pill {
    display: inline-block;
    background: rgba(0,194,168,0.12);
    border: 1px solid rgba(0,194,168,0.3);
    border-radius: 20px;
    padding: 3px 14px;
    font-family: 'DM Mono', monospace;
    font-size: .78rem;
    color: var(--accent);
    margin-bottom: .8rem;
}

/* Chat bubbles */
.bubble-user { background:#0d5c6b; color:#fff; border-radius:16px 16px 4px 16px; padding:.7rem 1rem; margin:.4rem 0 .4rem auto; max-width:80%; font-size:.9rem; }
.bubble-ai   { background:#f0f4f5; color:var(--text); border-radius:16px 16px 16px 4px; padding:.7rem 1rem; margin:.4rem auto .4rem 0; max-width:80%; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "patient_id": "patient_" + str(uuid.uuid4())[:8],
        "pipeline_result": None,
        "chat_agent": None,
        "chat_history": [],
        "upload_key": 0,           # increment to reset the file_uploader widget
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cv-header">
  <div>
    <h1>🏥 CareVision AI</h1>
    <p>Privacy-preserving multimodal clinical pipeline · DEEM Lab, Berlin</p>
  </div>
  <div class="cv-badge">GDPR Art. 17 Compliant</div>
  <div class="cv-badge">LangChain · Whisper · ChromaDB</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — audit log ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Audit Log")
    st.markdown(f'<div class="session-pill">🔑 {st.session_state.patient_id}</div>', unsafe_allow_html=True)

    try:
        with open(AUDIT_LOG_FILE) as f:
            logs = json.load(f)
        if logs:
            for entry in reversed(logs[-20:]):
                ts   = entry["timestamp"][11:19]
                atype = entry["action_type"]
                det  = entry["details"][:60]
                if "DELETION" in atype or "COMPLIANCE" in atype:
                    css = "log-delete"
                elif "ERROR" in atype or "FAIL" in entry.get("status", ""):
                    css = "log-error"
                else:
                    css = "log-success"
                st.markdown(f'<div class="{css}"><b>{ts}</b> {atype}<br>{det}</div>', unsafe_allow_html=True)
        else:
            st.caption("No logs yet.")
    except Exception:
        st.caption("Log file initialising…")

    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("""
- 🎙️ OpenAI Whisper (ASR)
- 👁️ OpenCV + scikit-learn (vision)
- 🦜 LangChain LCEL (orchestration)
- 🗄️ ChromaDB (vector store)
- 🔒 GDPR Art. 17 deletion pipeline
""")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_video, tab_chat, tab_voice, tab_search, tab_delete = st.tabs([
    "📹 Video Intake",
    "💬 Chat Agent",
    "🎙️ Voice Chat",
    "🔍 Semantic Search",
    "🗑️  Data Deletion",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Video Intake
# ════════════════════════════════════════════════════════════════════════════════
with tab_video:
    col_upload, col_results = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="cv-card"><h3>Upload Consultation Video</h3>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Accepts MP4 files",
            type=["mp4"],
            label_visibility="collapsed",
            key=f"video_upload_{st.session_state.upload_key}",
        )

        if uploaded:
            # Resolve this patient's own directories (creates them if needed)
            patient_dirs = get_patient_dirs(st.session_state.patient_id)
            file_path = os.path.join(patient_dirs["uploads"], uploaded.name)
            with open(file_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.video(file_path)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("▶ Run Full Pipeline", type="primary", key="run_pipeline"):
                log_action("PIPELINE_START", f"Processing video: {uploaded.name}")

                with st.status("Running multimodal pipeline…", expanded=True) as status:
                    st.write("🎙️ Extracting & transcribing audio (Whisper)…")
                    transcript = process_audio(file_path, patient_dirs["transcripts"])

                    st.write("🎞️ Extracting frames & running vision analysis (OpenCV + scikit-learn)…")
                    vision = process_video_frames(file_path, patient_dirs["frames"])

                    st.write("✅ Validating data quality…")
                    audio_path = os.path.join(patient_dirs["transcripts"], "extracted_audio.wav")
                    is_valid = validate_data(transcript, patient_dirs["frames"], audio_path)

                    if is_valid:
                        st.write("🦜 LangChain agent generating clinical summary…")
                        summary = generate_clinical_summary(transcript, vision)
                        store_embeddings(
                            st.session_state.patient_id,
                            transcript,
                            {
                                "frames": vision.get("frames_analyzed", 0),
                                "video_filename": uploaded.name,
                            },
                        )
                        st.session_state.pipeline_result = {
                            "transcript": transcript,
                            "vision": vision,
                            "summary": summary,
                        }
                        context = (
                            f"Transcript:\n{transcript}\n\n"
                            f"Vision:\n{json.dumps(vision, indent=2)}\n\n"
                            f"Summary:\n{json.dumps(summary, indent=2)}"
                        )
                        st.session_state.chat_agent = ClinicalChatAgent(context)
                        status.update(label="Pipeline complete ✓", state="complete")
                    else:
                        status.update(label="Validation failed", state="error")
                        st.error("Data validation failed — check the audit log for details.")
        else:
            st.markdown("*Upload an MP4 consultation video to begin.*")
            st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        result = st.session_state.pipeline_result
        if result:
            st.markdown('<div class="cv-card"><h3>📊 Vision Analysis</h3>', unsafe_allow_html=True)
            v = result["vision"]
            st.markdown(f"""
<div class="metric-row">
  <div class="metric-chip">Frames <span>{v.get('frames_analyzed', '—')}</span></div>
  <div class="metric-chip">Duration <span>{v.get('duration_seconds', '—')}s</span></div>
  <div class="metric-chip">Avg brightness <span>{v.get('avg_brightness', '—')}</span></div>
  <div class="metric-chip">Face presence <span>{v.get('avg_face_presence', '—')}</span></div>
  <div class="metric-chip">Anomalies <span>{v.get('anomalous_frame_count', '—')}</span></div>
</div>
<p style="font-size:.85rem;color:#5a7080">{v.get('visual_flags','')}</p>
<p style="font-size:.85rem"><b>Posture:</b> {v.get('posture_assessment','—')}</p>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="cv-card"><h3>🎙️ Transcript</h3>', unsafe_allow_html=True)
            st.text_area("", result["transcript"], height=110, label_visibility="collapsed")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="cv-card"><h3>🤖 Agent Clinical Summary</h3>', unsafe_allow_html=True)
            st.json(result["summary"])
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Run the pipeline to see results here.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Chat Agent
# ════════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Ask the Clinical Agent")
    st.caption("Query the processed patient record in natural language. Powered by LangChain + Mistral.")

    agent: ClinicalChatAgent | None = st.session_state.chat_agent

    if agent is None:
        st.info("Process a video first to activate the chat agent.")
    else:
        # Render history
        chat_container = st.container(height=380)
        with chat_container:
            for turn in st.session_state.chat_history:
                st.markdown(f'<div class="bubble-user">{turn["human"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bubble-ai">{turn["ai"]}</div>', unsafe_allow_html=True)

        user_input = st.chat_input("Ask about symptoms, observations, follow-up…")
        if user_input:
            reply = agent.chat(user_input)
            st.session_state.chat_history.append({"human": user_input, "ai": reply})
            st.rerun()

        col_a, col_b, col_c = st.columns(3)
        for col, q in zip(
            [col_a, col_b, col_c],
            [
                "What symptoms did the patient report?",
                "Were there any visual risk flags?",
                "What follow-up is recommended?",
            ],
        ):
            with col:
                if st.button(q, key=f"quick_{q[:10]}"):
                    reply = agent.chat(q)
                    st.session_state.chat_history.append({"human": q, "ai": reply})
                    st.rerun()


with tab_voice:
    agent_for_voice: ClinicalChatAgent | None = st.session_state.chat_agent
    if agent_for_voice is None:
        st.info("Process a video first to activate voice chat.")
    else:
        render_voice_chat_ui(agent_for_voice)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Semantic Search
# ════════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.markdown("### 🔍 Semantic Vector Search")
    st.caption("Search across all stored transcripts using ChromaDB cosine similarity.")

    query = st.text_input("Enter a clinical query:", placeholder="patient reports chest pain…")
    n_results = st.slider("Max results", 1, 5, 3)

    if st.button("Search", key="search_btn") and query:
        with st.spinner("Querying ChromaDB…"):
            hits = semantic_search(query, n_results)

        if hits:
            for i, h in enumerate(hits, 1):
                with st.expander(f"Result {i} — patient: {h['metadata'].get('patient_id', '?')}"):
                    st.write(h["document"][:400] + ("…" if len(h["document"]) > 400 else ""))
                    st.json(h["metadata"])
        else:
            st.warning("No results found. Process a video first to populate the vector store.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Data Deletion
# ════════════════════════════════════════════════════════════════════════════════
with tab_delete:
    st.markdown("### 🗑️ GDPR Article 17 — Right to be Forgotten")
    st.markdown("""
This pipeline executes a **verifiable, audited purge** of all patient data:

| Layer | What gets deleted |
|---|---|
| **Vector DB** | ChromaDB embeddings for this patient ID |
| **Audio** | Extracted WAV file from `/data/transcripts/` |
| **Video frames** | All JPEG frames from `/data/frames/` |
| **Source video** | Uploaded MP4 from `/data/uploads/` |
| **Transcript** | Plain-text `.txt` from `/data/transcripts/` |

Every deletion step is logged to the audit trail with a timestamp.
    """)

    st.error(
        f"⚠️ This action is **irreversible**. Target: `{st.session_state.patient_id}`",
        icon="🔴",
    )

    confirm = st.checkbox("I confirm I want to permanently delete all data for this patient.")
    if confirm:
        if st.button("Execute Right to be Forgotten", type="primary", key="delete_btn"):
            with st.spinner("Executing compliance purge…"):
                ok = execute_right_to_be_forgotten(st.session_state.patient_id)

            if ok:
                st.success("✅ All patient data purged. Audit trail preserved.")
            else:
                st.warning("Purge completed with some warnings — review the audit log.")

            # Reset session — upload_key increment forces file_uploader to remount empty
            st.session_state.patient_id      = "patient_" + str(uuid.uuid4())[:8]
            st.session_state.pipeline_result = None
            st.session_state.chat_agent      = None
            st.session_state.chat_history    = []
            st.session_state.upload_key     += 1
            st.rerun()