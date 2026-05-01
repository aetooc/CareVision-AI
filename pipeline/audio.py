import os
import numpy as np
import whisper
from moviepy import VideoFileClip
from pipeline.logging_utils import log_action


def process_audio(video_path: str, transcripts_dir: str) -> str:
    """
    Extracts audio from video and transcribes it using OpenAI Whisper.

    Pipeline:
      1. Extract WAV audio track via moviepy.
      2. Load Whisper 'base' model (fast; swap to 'small'/'medium' for accuracy).
      3. Transcribe with word-level timestamps.
      4. Persist transcript to disk for the deletion pipeline.

    Returns:
        Cleaned transcript string (empty string on failure).
    """
    audio_path = os.path.join(transcripts_dir, "extracted_audio.wav")
    transcript_path = os.path.join(transcripts_dir, "transcript.txt")

    # 1. Extract audio ──────────────────────────────────────────────────────────
    try:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            raise ValueError("Video file contains no audio track.")
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()
        log_action("DATA_EXTRACTION", "Audio track extracted from video successfully.")
    except Exception as e:
        log_action("ERROR", f"Audio extraction failed: {e}", status="FAILED")
        return ""

    # 2. Transcribe with Whisper ────────────────────────────────────────────────
    try:
        model = whisper.load_model("base")

        # word_timestamps=True gives us per-word timing for downstream use
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            language=None,          # auto-detect language
            verbose=False,
        )
        transcript_text = result["text"].strip()
        detected_lang = result.get("language", "unknown")

        # 3. Persist transcript ─────────────────────────────────────────────────
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        log_action(
            "AI_PROCESSING",
            f"Whisper transcription complete. "
            f"Language: {detected_lang}, Length: {len(transcript_text)} chars.",
        )
        return transcript_text

    except Exception as e:
        log_action("ERROR", f"Whisper transcription failed: {e}", status="FAILED")
        return ""


def get_audio_duration(audio_path: str) -> float:
    """Returns audio duration in seconds using moviepy (lightweight)."""
    try:
        from moviepy import AudioFileClip
        clip = AudioFileClip(audio_path)
        dur = clip.duration
        clip.close()
        return round(dur, 2)
    except Exception:
        return 0.0
