import os
import re
from pipeline.logging_utils import log_action

# ── DSPy setup ─────────────────────────────────────────────────────────────────

def _configure_dspy():
    """
    Configures DSPy with MistralAI if key is available.
    Returns True when a live LM is configured, False for demo/heuristic mode.
    """
    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        return False
    try:
        import dspy
        lm = dspy.LM(
            model="mistral/mistral-small-latest",
            api_key=api_key,
            temperature=0.2,
            max_tokens=512,
        )
        dspy.configure(lm=lm)
        return True
    except Exception as e:
        log_action("WARNING", f"DSPy configuration failed: {e}")
        return False


# ── Signatures ─────────────────────────────────────────────────────────────────

def _get_severity_module():
    """Lazy-import DSPy and return the ChainOfThought severity module."""
    import dspy

    class SeveritySignature(dspy.Signature):
        """
        You are a clinical triage assistant.
        Given a patient transcript and visual observations from a telemedicine session,
        classify the overall clinical severity and provide a brief reasoning.
        """
        transcript: str = dspy.InputField(desc="Patient's spoken transcript from the consultation")
        visual_observations: str = dspy.InputField(desc="Visual/postural observations from frame analysis")
        severity: str = dspy.OutputField(desc="One of: low | medium | high | critical")
        reasoning: str = dspy.OutputField(desc="2-3 sentence clinical rationale for the severity rating")
        triage_priority: str = dspy.OutputField(desc="Recommended triage action, e.g. 'Routine follow-up in 7 days'")

    return dspy.ChainOfThought(SeveritySignature)


def _get_risk_module():
    """Lazy-import DSPy and return the Predict risk-flag module."""
    import dspy

    class RiskSignature(dspy.Signature):
        """
        Extract structured risk signals from a telemedicine consultation transcript.
        Focus on: medication mentions, pain descriptors, mental health indicators,
        chronic condition references, and urgent symptoms.
        """
        transcript: str = dspy.InputField(desc="Full consultation transcript")
        risk_flags: str = dspy.OutputField(
            desc="Comma-separated list of risk signals found, or 'none' if clean"
        )
        urgent: str = dspy.OutputField(desc="'yes' or 'no' — does this require same-day clinical review?")
        confidence: str = dspy.OutputField(desc="Confidence in extraction: low | medium | high")

    return dspy.Predict(RiskSignature)


# ── Heuristic fallback (demo mode) ─────────────────────────────────────────────

_HIGH_RISK_KEYWORDS = [
    "chest pain", "can't breathe", "cannot breathe", "shortness of breath",
    "stroke", "unconscious", "bleeding", "suicidal", "severe pain",
    "heart", "collapsed", "emergency",
]
_MEDIUM_RISK_KEYWORDS = [
    "pain", "fever", "dizzy", "dizziness", "fatigue", "vomiting",
    "nausea", "infection", "swelling", "medication", "medication change",
    "anxious", "anxiety", "depressed", "depression",
]

def _heuristic_severity(transcript: str) -> dict:
    """Rule-based fallback when no API key is present."""
    lower = transcript.lower()
    if any(kw in lower for kw in _HIGH_RISK_KEYWORDS):
        severity, priority = "high", "Same-day clinical review recommended."
    elif any(kw in lower for kw in _MEDIUM_RISK_KEYWORDS):
        severity, priority = "medium", "Follow-up within 48–72 hours."
    else:
        severity, priority = "low", "Routine follow-up in 7 days."

    flags = [kw for kw in _MEDIUM_RISK_KEYWORDS + _HIGH_RISK_KEYWORDS if kw in lower]
    return {
        "severity": severity,
        "reasoning": f"Heuristic analysis (demo mode — set MISTRAL_API_KEY for DSPy). "
                     f"Keywords matched: {', '.join(flags[:5]) if flags else 'none'}.",
        "triage_priority": priority,
        "risk_flags": flags[:8] if flags else ["none"],
        "urgent": "yes" if severity in ("high", "critical") else "no",
        "confidence": "low",
        "dspy_active": False,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def run_dspy_analysis(transcript: str, vision_analysis: dict) -> dict:
    """
    Main entry point. Runs both DSPy modules (or heuristic fallback) and
    returns a unified dict consumed by app.py for display.

    Args:
        transcript:      Full Whisper transcript string.
        vision_analysis: Dict from process_video_frames() — needs
                         'visual_flags' and 'posture_assessment' keys.

    Returns dict with keys:
        severity, reasoning, triage_priority,
        risk_flags (list), urgent (bool), confidence, dspy_active (bool)
    """
    if not transcript or len(transcript.strip()) < 10:
        log_action("DSPY_ANALYSIS", "Transcript too short — skipping DSPy analysis.")
        return _heuristic_severity(transcript or "")

    visual_obs = (
        vision_analysis.get("visual_flags", "No visual data.")
        + " "
        + vision_analysis.get("posture_assessment", "")
    ).strip()

    live = _configure_dspy()

    if not live:
        result = _heuristic_severity(transcript)
        log_action("DSPY_ANALYSIS", "Demo mode — heuristic severity used.")
        return result

    try:
        # Module 1: ChainOfThought severity
        severity_mod = _get_severity_module()
        sev_result = severity_mod(
            transcript=transcript[:1500],   # token-safe truncation
            visual_observations=visual_obs[:500],
        )

        # Module 2: Predict risk flags
        risk_mod = _get_risk_module()
        risk_result = risk_mod(transcript=transcript[:1500])

        # Parse risk_flags string → list
        raw_flags = risk_result.risk_flags or "none"
        flags_list = (
            [f.strip() for f in raw_flags.split(",") if f.strip().lower() != "none"]
            if raw_flags.lower() != "none"
            else []
        )

        output = {
            "severity": sev_result.severity.strip().lower(),
            "reasoning": sev_result.reasoning.strip(),
            "triage_priority": sev_result.triage_priority.strip(),
            "risk_flags": flags_list,
            "urgent": risk_result.urgent.strip().lower() == "yes",
            "confidence": risk_result.confidence.strip().lower(),
            "dspy_active": True,
        }

        log_action(
            "DSPY_ANALYSIS",
            f"DSPy ChainOfThought complete. Severity={output['severity']}, "
            f"Urgent={output['urgent']}, Flags={len(flags_list)}.",
        )
        return output

    except Exception as e:
        log_action("ERROR", f"DSPy pipeline failed, falling back to heuristic: {e}", status="FAILED")
        result = _heuristic_severity(transcript)
        result["dspy_active"] = False
        return result


def format_severity_badge(severity: str) -> tuple[str, str]:
    """
    Returns (emoji, colour_hex) for Streamlit st.markdown badge rendering.
    Usage: emoji, colour = format_severity_badge(dspy_result['severity'])
    """
    mapping = {
        "low":      ("🟢", "#2d6a4f"),
        "medium":   ("🟡", "#b5830a"),
        "high":     ("🔴", "#9b2226"),
        "critical": ("🚨", "#6a0572"),
    }
    return mapping.get(severity.lower(), ("⚪", "#555555"))
