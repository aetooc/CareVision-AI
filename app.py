import os
import streamlit as st

st.set_page_config(
    page_title="CareVision AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


for key in ("MISTRAL_API_KEY",):
    if key in st.secrets and not os.getenv(key):
        os.environ[key] = st.secrets[key]

from pipeline.session      import (
    init_session, render_session_controls, get_patient_id,
    mark_upload_done,
    KEY_PIPELINE_RESULT, KEY_AGENT, KEY_CHAT_MESSAGES, KEY_DSPY_RESULT,
)
from pipeline.storage      import get_patient_dirs, store_embeddings, semantic_search, execute_right_to_be_forgotten
from pipeline.audio        import process_audio
from pipeline.video        import process_video_frames
from pipeline.validation   import validate_data
from pipeline.agent        import generate_clinical_summary, ClinicalChatAgent
from pipeline.dspy_module  import run_dspy_analysis, format_severity_badge
from pipeline.logging_utils import log_action
from pipeline.voice        import render_voice_chat_ui, synthesize_speech

init_session()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📋 Audit Log")

    # Patient session controls (new widget)
    render_session_controls()

    # Audit log display
    from pipeline.logging_utils import AUDIT_LOG_FILE
    import json

    try:
        with open(AUDIT_LOG_FILE, "r") as f:
            logs = json.load(f)
    except Exception:
        logs = []

    patient_id = get_patient_id()

    # Filter to current patient if logs exist
    patient_logs = [l for l in logs if patient_id in l.get("details", "")]
    if not patient_logs:
        patient_logs = logs[-20:]  # show last 20 global if no patient match

    st.markdown(
        f"<div style='background:#1a1a2e;border:1px solid #2d6a4f;"
        f"border-radius:6px;padding:6px 10px;margin-bottom:6px;'>"
        f"<code style='color:#74c69d;font-size:11px;'>{patient_id}</code>"
        f"<span style='color:#6c757d;font-size:10px;'> · {len(patient_logs)} events</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    ACTION_COLOURS = {
        "COMPLIANCE_SUCCESS": "#2d6a4f",
        "DATA_DELETION":      "#9b2226",
        "VALIDATION_ERROR":   "#9b2226",
        "ERROR":              "#9b2226",
        "AI_PROCESSING":      "#1d4e89",
        "DATA_EXTRACTION":    "#495057",
        "DATA_STORAGE":       "#495057",
        "DATA_VALIDATION":    "#495057",
        "DSPY_ANALYSIS":      "#5c4d7d",
        "VOICE_TTS":          "#4a4e69",
        "VOICE_STT":          "#4a4e69",
        "CHAT_AGENT":         "#2b4162",
    }

    for entry in reversed(patient_logs[-30:]):
        colour = ACTION_COLOURS.get(entry["action_type"], "#333")
        st.markdown(
            f"""<div style="border-left:3px solid {colour};
                padding:6px 8px;margin-bottom:4px;background:#1a1a1a;
                border-radius:0 4px 4px 0;">
                <span style="color:#adb5bd;font-size:10px;">
                    {entry['timestamp'][11:19]}</span>
                <span style="color:#e9ecef;font-size:10px;font-weight:600;">
                    &nbsp;{entry['action_type']}</span><br/>
                <span style="color:#6c757d;font-size:10px;">
                    {entry['details'][:80]}</span>
            </div>""",
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#0d4f4f,#0a3a3a);
         padding:24px 32px;border-radius:12px;margin-bottom:24px;">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;">
        <div>
          <h1 style="margin:0;color:#ffffff;font-size:2rem;">🏥 CareVision AI</h1>
          <p style="margin:4px 0 0;color:#a8d8d8;font-size:0.95rem;">
            Privacy-preserving multimodal clinical pipeline
          </p>
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <span style="background:#0a5c5c;color:#7dffd6;padding:4px 12px;
                border-radius:20px;font-size:12px;font-weight:600;border:1px solid #1a8a8a;">
            GDPR Art. 17 Compliant</span>
          <span style="background:#0a3a5c;color:#7dc8ff;padding:4px 12px;
                border-radius:20px;font-size:12px;font-weight:600;border:1px solid #1a6a9a;">
            LangChain · Whisper · ChromaDB</span>
          <span style="background:#3a0a5c;color:#d4a7ff;padding:4px 12px;
                border-radius:20px;font-size:12px;font-weight:600;border:1px solid #7a2aaa;">
            DSPy ChainOfThought</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_intake, tab_dspy, tab_chat, tab_voice, tab_search, tab_delete = st.tabs([
    "📹 Video Intake",
    "🧠 DSPy Analysis",
    "💬 Chat Agent",
    "🎙️ Voice Chat",
    "🔍 Semantic Search",
    "🗑️ Data Deletion",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — VIDEO INTAKE
# ─────────────────────────────────────────────────────────────────────────────
with tab_intake:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("### Upload Consultation Video")
        uploaded_file = st.file_uploader(
            label="",
            type=["mp4"],
            help="200MB per file · MP4",
            key=f"uploader_{get_patient_id()}",   # key tied to patient so it resets
        )
        if uploaded_file:
            st.caption("_Upload an MP4 consultation video to begin. "
                       "Video is processed locally and never transmitted externally._")

    with col_right:
        pipeline_result = st.session_state.get(KEY_PIPELINE_RESULT)

        if not pipeline_result:
            st.markdown(
                "<div style='background:#0d1f2d;border:1px solid #1d4e89;"
                "border-radius:8px;padding:24px;color:#6c9fc4;font-size:0.95rem;'>"
                "Run the pipeline to see results here.<br/><br/>"
                "<small style='color:#495057;'>Upload an MP4 above, then click "
                "<strong>Run Pipeline</strong>.</small></div>",
                unsafe_allow_html=True,
            )
        else:
            summary = pipeline_result.get("clinical_summary", {})
            st.markdown("#### 📊 Clinical Summary")
            st.json(summary)

    # Run pipeline button
    if uploaded_file:
        if st.button("▶️  Run Pipeline", type="primary", use_container_width=True):
            patient_id = get_patient_id()
            dirs = get_patient_dirs(patient_id)

            # Save upload
            video_path = os.path.join(dirs["uploads"], uploaded_file.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.read())
            log_action("DATA_EXTRACTION", f"Video saved for {patient_id}.")

            with st.spinner("🎞️ Extracting video frames…"):
                vision = process_video_frames(video_path, dirs["frames"])

            with st.spinner("🎙️ Transcribing audio (Whisper)…"):
                transcript = process_audio(video_path, dirs["transcripts"])

            audio_path = os.path.join(dirs["transcripts"], "extracted_audio.wav")
            valid = validate_data(transcript, dirs["frames"], audio_path)

            if not valid:
                st.error("Validation failed — check the audit log.")
            else:
                with st.spinner("🤖 Generating clinical summary (LangChain + Mistral)…"):
                    summary = generate_clinical_summary(transcript, vision)

                with st.spinner("🧠 Running DSPy severity analysis…"):
                    dspy_result = run_dspy_analysis(transcript, vision)
                    st.session_state[KEY_DSPY_RESULT] = dspy_result

                store_embeddings(patient_id, transcript, metadata={
                    "frames": vision.get("frames_analyzed", 0),
                    "video":  uploaded_file.name,
                })

                result = {
                    "transcript":       transcript,
                    "vision_analysis":  vision,
                    "clinical_summary": summary,
                }
                st.session_state[KEY_PIPELINE_RESULT] = result

                # Build context string for chat agent
                context = (
                    f"Patient ID: {patient_id}\n"
                    f"Transcript: {transcript[:800]}\n"
                    f"Visual: {vision.get('visual_flags','')}\n"
                    f"Posture: {vision.get('posture_assessment','')}\n"
                    f"DSPy Severity: {dspy_result.get('severity','')}\n"
                    f"DSPy Triage: {dspy_result.get('triage_priority','')}\n"
                    f"Summary: {summary.get('summary','')}"
                )
                st.session_state[KEY_AGENT] = ClinicalChatAgent(context)
                mark_upload_done()

                st.success("✅ Pipeline complete! See DSPy Analysis tab for severity assessment.")
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DSPY ANALYSIS (new)
# ─────────────────────────────────────────────────────────────────────────────
with tab_dspy:
    st.markdown("### 🧠 DSPy Clinical Intelligence")
    st.caption(
        "Powered by DSPy `ChainOfThought` — the model's reasoning steps are "
        "fully visible below for auditability."
    )

    dspy_result = st.session_state.get(KEY_DSPY_RESULT)

    if not dspy_result:
        st.info("Run the video intake pipeline first to see DSPy analysis.")
    else:
        severity  = dspy_result.get("severity", "unknown")
        emoji, colour = format_severity_badge(severity)

        # Severity banner
        st.markdown(
            f"""<div style="background:{colour}22;border:2px solid {colour};
                border-radius:10px;padding:16px 24px;margin-bottom:16px;">
                <span style="font-size:2rem;">{emoji}</span>
                <span style="color:#ffffff;font-size:1.4rem;font-weight:700;
                    margin-left:12px;text-transform:uppercase;">{severity} severity</span>
                <br/>
                <span style="color:#adb5bd;font-size:0.9rem;margin-left:4px;">
                    {dspy_result.get('triage_priority','')}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🔍 ChainOfThought Reasoning")
            st.markdown(
                f"<div style='background:#1a1a2e;border-radius:8px;padding:12px;"
                f"color:#c5d8f0;font-size:0.9rem;line-height:1.6;'>"
                f"{dspy_result.get('reasoning','No reasoning available.')}"
                f"</div>",
                unsafe_allow_html=True,
            )

            dspy_active = dspy_result.get("dspy_active", False)
            mode_label = "⚡ DSPy + Mistral (live)" if dspy_active else "🟡 Heuristic (demo mode)"
            confidence = dspy_result.get("confidence", "n/a")
            st.caption(f"Mode: {mode_label} · Confidence: {confidence}")

        with col2:
            st.markdown("#### 🚩 Risk Flags")
            flags = dspy_result.get("risk_flags", [])
            urgent = dspy_result.get("urgent", False)

            if urgent:
                st.error("⚠️ Urgent — same-day clinical review recommended")

            if flags:
                for flag in flags:
                    st.markdown(
                        f"<span style='background:#2d1b1b;color:#ff8585;"
                        f"padding:3px 10px;border-radius:12px;font-size:12px;"
                        f"margin:2px;display:inline-block;'>{flag}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.success("No significant risk flags detected.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — CHAT AGENT
# ─────────────────────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown("### 💬 Clinical Chat Agent")

    agent = st.session_state.get(KEY_AGENT)
    if not agent:
        st.info("Run the video intake pipeline first to activate the chat agent.")
    else:
        messages = st.session_state.get(KEY_CHAT_MESSAGES, [])
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about this consultation…"):
            messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    reply = agent.chat(prompt)
                st.markdown(reply)
            messages.append({"role": "assistant", "content": reply})
            st.session_state[KEY_CHAT_MESSAGES] = messages

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — VOICE CHAT
# ─────────────────────────────────────────────────────────────────────────────
with tab_voice:
    agent = st.session_state.get(KEY_AGENT)
    if not agent:
        st.info("Run the video intake pipeline first to activate voice chat.")
    else:
        render_voice_chat_ui(agent)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — SEMANTIC SEARCH
# ─────────────────────────────────────────────────────────────────────────────
with tab_search:
    st.markdown("### 🔍 Semantic Search")
    st.caption("Search across all stored patient transcripts using cosine similarity (ChromaDB).")

    query = st.text_input("Enter clinical query:", placeholder="e.g. patient reports chest pain")
    n_results = st.slider("Max results", 1, 10, 3)

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching embeddings…"):
            hits = semantic_search(query, n_results=n_results)
        if hits:
            for i, hit in enumerate(hits, 1):
                with st.expander(f"Result {i} — patient `{hit['metadata'].get('patient_id','?')}`"):
                    st.write(hit["document"][:500])
                    st.json(hit["metadata"])
        else:
            st.warning("No results found.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — DATA DELETION (GDPR Art. 17)
# ─────────────────────────────────────────────────────────────────────────────
with tab_delete:
    st.markdown("### 🗑️ Right to be Forgotten — GDPR Article 17")
    st.warning(
        "This action permanently deletes all data for the specified patient "
        "from ChromaDB, uploads, frames, and transcripts directories. "
        "This cannot be undone."
    )

    del_id = st.text_input(
        "Patient ID to delete:",
        value=get_patient_id(),
        help="Defaults to the current active patient.",
    )

    confirm_del = st.checkbox("I confirm this deletion is authorised and irreversible.")

    if st.button("🗑️ Execute Deletion", type="primary", disabled=not confirm_del):
        if del_id:
            with st.spinner(f"Purging all data for {del_id}…"):
                ok = execute_right_to_be_forgotten(del_id)
            if ok:
                st.success(f"✅ COMPLIANCE_SUCCESS — all data for `{del_id}` has been purged.")
                # If we just deleted the current patient, start a new session
                if del_id == get_patient_id():
                    from pipeline.session import start_new_patient
                    start_new_patient()
                    st.info("Current session cleared. New patient session started.")
                    st.rerun()
            else:
                st.error("Deletion partially failed — check the audit log for details.")
        else:
            st.error("Please enter a patient ID.")
