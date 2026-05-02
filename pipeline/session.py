import uuid
from datetime import datetime


# Using string constants avoids typo bugs across files.
KEY_PATIENT_ID       = "patient_id"
KEY_SESSION_HISTORY  = "session_history"    # list of past {id, started_at} dicts
KEY_PIPELINE_RESULT  = "pipeline_result"    # dict returned by the processing pipeline
KEY_AGENT            = "chat_agent"         # ClinicalChatAgent instance
KEY_CHAT_MESSAGES    = "chat_messages"      # list of {role, content} for display
KEY_DSPY_RESULT      = "dspy_result"        # dict from run_dspy_analysis()
KEY_UPLOAD_DONE      = "upload_done"        # bool — True after video processed


def _new_patient_id() -> str:
    """Generates a short, human-readable patient session ID."""
    return "patient_" + uuid.uuid4().hex[:8]


def init_session() -> None:
    """
    Must be called once at the top of app.py (before any st.* calls).
    Initialises all session_state keys if they don't already exist.
    Safe to call multiple times — idempotent.
    """
    import streamlit as st

    if KEY_PATIENT_ID not in st.session_state:
        st.session_state[KEY_PATIENT_ID] = _new_patient_id()

    if KEY_SESSION_HISTORY not in st.session_state:
        st.session_state[KEY_SESSION_HISTORY] = []

    for key, default in [
        (KEY_PIPELINE_RESULT, None),
        (KEY_AGENT,           None),
        (KEY_CHAT_MESSAGES,   []),
        (KEY_DSPY_RESULT,     None),
        (KEY_UPLOAD_DONE,     False),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


def get_patient_id() -> str:
    """Returns the active patient ID. Call after init_session()."""
    import streamlit as st
    return st.session_state.get(KEY_PATIENT_ID, _new_patient_id())


def start_new_patient() -> str:
    """
    Archives the current patient to session history, then wipes all
    pipeline state for a fresh intake — without a page reload.

    Returns the new patient ID.
    """
    import streamlit as st

    current_id = st.session_state.get(KEY_PATIENT_ID)

    # Archive to history if this session had any real data
    if current_id and st.session_state.get(KEY_UPLOAD_DONE):
        history = st.session_state.get(KEY_SESSION_HISTORY, [])
        history.append({
            "id":         current_id,
            "started_at": datetime.now().strftime("%H:%M:%S"),
            "had_data":   True,
        })
        st.session_state[KEY_SESSION_HISTORY] = history

    # Generate fresh ID and reset all pipeline state
    new_id = _new_patient_id()
    st.session_state[KEY_PATIENT_ID]      = new_id
    st.session_state[KEY_PIPELINE_RESULT] = None
    st.session_state[KEY_AGENT]           = None
    st.session_state[KEY_CHAT_MESSAGES]   = []
    st.session_state[KEY_DSPY_RESULT]     = None
    st.session_state[KEY_UPLOAD_DONE]     = False

    return new_id


def mark_upload_done() -> None:
    """Call after the pipeline finishes processing a video."""
    import streamlit as st
    st.session_state[KEY_UPLOAD_DONE] = True


def render_session_controls() -> None:
    """
    Renders the patient session widget inside the Streamlit sidebar.
    Shows:
      - Current patient ID badge
      - "New Patient" button (clears pipeline, generates new ID)
      - Collapsible session history for the current browser session
    
    Call this inside the sidebar block in app.py:
        with st.sidebar:
            render_session_controls()
            # ... rest of sidebar (audit log, etc.)
    """
    import streamlit as st

    patient_id  = get_patient_id()
    upload_done = st.session_state.get(KEY_UPLOAD_DONE, False)
    history     = st.session_state.get(KEY_SESSION_HISTORY, [])

    st.markdown("---")
    st.markdown("### 👤 Patient Session")

    # Current patient badge
    st.markdown(
        f"""
        <div style="
            background: #1a3a2a;
            border: 1px solid #2d6a4f;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 8px;
        ">
            <span style="color:#74c69d; font-size:11px; font-weight:600;">ACTIVE PATIENT</span><br/>
            <code style="color:#d8f3dc; font-size:13px;">{patient_id}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status indicator
    if upload_done:
        st.markdown(
            "<span style='color:#74c69d; font-size:12px;'>✅ Consultation recorded</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<span style='color:#adb5bd; font-size:12px;'>⏳ Awaiting video upload</span>",
            unsafe_allow_html=True,
        )

    st.write("")

    # New Patient button — confirm if current session has data to avoid accidents
    if upload_done:
        confirm = st.checkbox("Confirm: start new patient (current data stays in DB)", key="confirm_new_patient")
        btn_disabled = not confirm
    else:
        btn_disabled = False

    if st.button(
        "➕  New Patient",
        use_container_width=True,
        disabled=btn_disabled,
        type="primary",
        key="btn_new_patient",
    ):
        new_id = start_new_patient()
        st.success(f"New session started: `{new_id}`")
        st.rerun()

    # Session history (collapsible)
    if history:
        with st.expander(f"📋 Session history ({len(history)} patients)", expanded=False):
            for entry in reversed(history):
                st.markdown(
                    f"<code style='font-size:11px;'>{entry['id']}</code> "
                    f"<span style='color:#6c757d; font-size:11px;'>— {entry['started_at']}</span>",
                    unsafe_allow_html=True,
                )
    st.markdown("---")
