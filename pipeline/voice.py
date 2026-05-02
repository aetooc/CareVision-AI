"""
voice.py  —  Real-time voice interface for CareVision
Adds STT (speech → text) + TTS (text → speech) to the ClinicalChatAgent.

Dependencies:
    pip install SpeechRecognition gTTS pydub sounddevice numpy
    # or for Streamlit Cloud: use openai-whisper (already a dep) for STT
    #                          and gTTS for TTS

Two modes:
  - Microphone capture  (local / desktop Streamlit)
  - Uploaded audio file (Streamlit Cloud / demo mode)
"""

import io
import os
import tempfile
import numpy as np
from pipeline.logging_utils import log_action


# ── Text-to-Speech ─────────────────────────────────────────────────────────────

def synthesize_speech(text: str, lang: str = "en") -> bytes:
    """
    Converts text to MP3 bytes using gTTS.
    Returns raw MP3 bytes that Streamlit can play with st.audio().

    Usage in app.py:
        audio_bytes = synthesize_speech(agent_reply)
        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        log_action("VOICE_TTS", f"Synthesized {len(text)} chars → MP3 ({lang}).")
        return buf.read()
    except ImportError:
        log_action("WARNING", "gTTS not installed. Run: pip install gTTS")
        return b""
    except Exception as e:
        log_action("ERROR", f"TTS synthesis failed: {e}", status="FAILED")
        return b""


# ── Speech-to-Text (uploaded audio file) ──────────────────────────────────────

def transcribe_voice_input(audio_bytes: bytes, file_ext: str = "wav") -> str:
    """
    Transcribes a voice message (from st.audio_input or file_uploader) using
    OpenAI Whisper (already a CareVision dependency).

    Args:
        audio_bytes: raw bytes from Streamlit's audio widget
        file_ext:    'wav', 'mp3', 'ogg', 'm4a'

    Returns:
        Transcribed text string, or "" on failure.

    Usage in app.py:
        audio_val = st.audio_input("Speak your question")
        if audio_val:
            question = transcribe_voice_input(audio_val.read(), "wav")
            reply = agent.chat(question)
    """
    try:
        import whisper

        # Write bytes to a temp file (Whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = whisper.load_model("base")
        result = model.transcribe(tmp_path, language=None, verbose=False)
        text = result["text"].strip()

        os.unlink(tmp_path)
        log_action("VOICE_STT", f"Voice input transcribed: {len(text)} chars.")
        return text

    except ImportError:
        log_action("WARNING", "openai-whisper not installed.")
        return ""
    except Exception as e:
        log_action("ERROR", f"Voice STT failed: {e}", status="FAILED")
        return ""


# ── Streamlit UI component ─────────────────────────────────────────────────────

def render_voice_chat_ui(agent) -> None:
    """
    Drop-in Streamlit component for voice-enabled chat.
    Renders mic input + text fallback + TTS playback.

    Call this inside an st.tab or st.expander in app.py:
        from pipeline.voice import render_voice_chat_ui
        with st.expander("🎙️ Voice Chat"):
            render_voice_chat_ui(agent)
    """
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown("**Voice consultation assistant**")
    st.caption("Speak a clinical question or type below. The agent will reply in text and audio.")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Streamlit ≥ 1.31 has st.audio_input (experimental)
        voice_input = None
        try:
            voice_input = st.audio_input("Record your question", key="voice_rec")
        except AttributeError:
            st.info("Upgrade Streamlit ≥ 1.31 for microphone capture. Using text input instead.")

        text_input = st.text_input("Or type your question:", key="voice_text", placeholder="e.g. What were the main symptoms reported?")

    with col2:
        st.write("")
        st.write("")
        submit = st.button("Send ↗", key="voice_send", use_container_width=True)

    if submit:
        user_message = ""

        # Prefer voice if available
        if voice_input is not None:
            with st.spinner("Transcribing voice..."):
                user_message = transcribe_voice_input(voice_input.read(), "wav")
            if user_message:
                st.markdown(f"**You said:** *{user_message}*")
            else:
                st.warning("Could not transcribe. Using text input.")

        if not user_message and text_input:
            user_message = text_input

        if user_message:
            with st.spinner("Agent thinking..."):
                reply = agent.chat(user_message)

            st.markdown(f"**Agent:** {reply}")

            # TTS playback
            with st.spinner("Generating audio response..."):
                audio_bytes = synthesize_speech(reply)

            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
            else:
                st.caption("(Audio synthesis unavailable — install gTTS)")
        else:
            st.warning("Please speak or type a question first.")
