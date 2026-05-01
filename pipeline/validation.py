import os
from pipeline.logging_utils import log_action


def validate_data(transcript: str, frames_dir: str, audio_path: str) -> bool:
    """
    Quality-gate before the AI pipeline runs.

    Checks:
      - Transcript is non-empty and long enough to be meaningful.
      - At least one video frame was extracted.
      - Audio file exists and is larger than a noise floor threshold.
    Returns True only when all checks pass.
    """
    try:
        # 1. Transcript ────────────────────────────────────────────────────────
        if not transcript or len(transcript.strip()) < 10:
            raise ValueError(
                f"Transcript too short ({len(transcript.strip())} chars). "
                "Check audio quality or Whisper model size."
            )

        # 2. Video frames ──────────────────────────────────────────────────────
        frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")]
        if not frame_files:
            raise ValueError("No JPEG frames found – video extraction may have failed.")

        # 3. Audio file ────────────────────────────────────────────────────────
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Expected audio at {audio_path} but file is missing.")

        audio_size = os.path.getsize(audio_path)
        if audio_size < 1_000:          # < 1 KB → clearly corrupt
            raise ValueError(
                f"Audio file suspiciously small ({audio_size} bytes). "
                "Possible corrupt extraction."
            )

        log_action(
            "DATA_VALIDATION",
            f"All checks passed — transcript={len(transcript)} chars, "
            f"frames={len(frame_files)}, audio={audio_size:,} bytes.",
        )
        return True

    except Exception as e:
        log_action("VALIDATION_ERROR", str(e), status="FAILED")
        return False
