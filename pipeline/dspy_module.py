import os
import re
import traceback
from pipeline.logging_utils import log_action

# -- Debug trace collector -----------------------------------------------------
dspy_debug_trace: list[dict] = []

def _trace(stage: str, status: str, message: str, exc: Exception | None = None) -> None:
    """Append a structured debug entry and log it."""
    entry = {
        "stage":   stage,
        "status":  status,
        "message": message,
        "detail":  traceback.format_exc() if exc else "",
    }
    dspy_debug_trace.append(entry)
    level = "ERROR" if status == "error" else "WARNING" if status == "warn" else "DSPY_ANALYSIS"
    log_action(level, f"[{stage}] {message}")


def get_debug_trace() -> list[dict]:
    return list(dspy_debug_trace)


def clear_debug_trace() -> None:
    dspy_debug_trace.clear()


# -- DSPy setup ----------------------------------------------------------------

def _build_lm():
    """
    Imports dspy/litellm and instantiates a dspy.LM object.

    Does NOT call dspy.configure() -- callers must use dspy.context(lm=lm)
    so each Streamlit thread gets its own settings without causing:
      RuntimeError: dspy.settings can only be changed by the thread that
      initially configured it.

    Returns: (lm | None, reason_str)
    """
    # Step 1: API key
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        reason = (
            "MISTRAL_API_KEY is not set or is empty. "
            "Add it to .streamlit/secrets.toml as: MISTRAL_API_KEY = \"your_key\". "
            "Also confirm app.py copies secrets to os.environ before this module is called."
        )
        _trace("API_KEY_CHECK", "error", reason)
        return None, reason

    if len(api_key) < 10:
        reason = f"MISTRAL_API_KEY looks malformed (length={len(api_key)}). Check for copy-paste errors."
        _trace("API_KEY_CHECK", "warn", reason)
        return None, reason

    _trace("API_KEY_CHECK", "ok", f"API key present (length={len(api_key)}, prefix={api_key[:6]}...)")

    # Step 2: Import dspy
    try:
        import dspy
        _trace("DSPY_IMPORT", "ok", f"dspy imported successfully (version={getattr(dspy, '__version__', 'unknown')})")
    except ImportError as e:
        reason = f"Cannot import dspy: {e}. Run: pip install dspy"
        _trace("DSPY_IMPORT", "error", reason, exc=e)
        return None, reason
    except Exception as e:
        reason = f"Unexpected error importing dspy: {type(e).__name__}: {e}"
        _trace("DSPY_IMPORT", "error", reason, exc=e)
        return None, reason

    # Step 3: Import litellm
    try:
        import litellm
        _trace("LITELLM_IMPORT", "ok", f"litellm imported (version={getattr(litellm, '__version__', 'unknown')})")
    except ImportError as e:
        reason = f"Cannot import litellm: {e}. Run: pip install litellm>=1.0"
        _trace("LITELLM_IMPORT", "error", reason, exc=e)
        return None, reason
    except Exception as e:
        reason = f"Unexpected error importing litellm: {type(e).__name__}: {e}"
        _trace("LITELLM_IMPORT", "error", reason, exc=e)
        return None, reason

    # Step 4: Instantiate dspy.LM (no global configure -- done per-call via dspy.context)
    try:
        lm = dspy.LM(
            model="mistral/mistral-small-latest",
            api_key=api_key,
            temperature=0.2,
            max_tokens=512,
        )
        _trace("DSPY_LM_INIT", "ok", "dspy.LM instantiated with mistral/mistral-small-latest")
    except TypeError as e:
        reason = (
            f"dspy.LM() raised TypeError: {e}. "
            "This usually means your dspy version has a different constructor signature. "
            f"dspy version: {getattr(dspy, '__version__', 'unknown')}. "
            "Try: pip install --upgrade dspy"
        )
        _trace("DSPY_LM_INIT", "error", reason, exc=e)
        return None, reason
    except Exception as e:
        reason = (
            f"dspy.LM() failed with {type(e).__name__}: {e}. "
            "Possible causes: invalid model string, network blocked on Streamlit Cloud, "
            "or litellm/mistral provider issue."
        )
        _trace("DSPY_LM_INIT", "error", reason, exc=e)
        return None, reason

    _trace("DSPY_CONFIGURE", "ok",
           "dspy.LM ready -- will activate via dspy.context(lm=lm) per-call (thread-safe)")
    return lm, "DSPy LM built successfully with Mistral."


# -- Signatures ----------------------------------------------------------------

def _get_severity_module():
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
        urgent: str = dspy.OutputField(desc="'yes' or 'no' -- does this require same-day clinical review?")
        confidence: str = dspy.OutputField(desc="Confidence in extraction: low | medium | high")

    return dspy.Predict(RiskSignature)


# -- Heuristic fallback (demo mode) --------------------------------------------

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

def _heuristic_severity(transcript: str, fallback_reason: str = "") -> dict:
    lower = transcript.lower()
    if any(kw in lower for kw in _HIGH_RISK_KEYWORDS):
        severity, priority = "high", "Same-day clinical review recommended."
    elif any(kw in lower for kw in _MEDIUM_RISK_KEYWORDS):
        severity, priority = "medium", "Follow-up within 48-72 hours."
    else:
        severity, priority = "low", "Routine follow-up in 7 days."

    flags = [kw for kw in _MEDIUM_RISK_KEYWORDS + _HIGH_RISK_KEYWORDS if kw in lower]
    return {
        "severity": severity,
        "reasoning": (
            "Heuristic analysis (demo mode -- set MISTRAL_API_KEY for DSPy). "
            f"Keywords matched: {', '.join(flags[:5]) if flags else 'none'}."
        ),
        "triage_priority": priority,
        "risk_flags": flags[:8] if flags else ["none"],
        "urgent": "yes" if severity in ("high", "critical") else "no",
        "confidence": "low",
        "dspy_active": False,
        "debug_fallback_reason": fallback_reason,
        "debug_trace": get_debug_trace(),
    }


# -- Public API ----------------------------------------------------------------

def run_dspy_analysis(transcript: str, vision_analysis: dict) -> dict:
    """
    Main entry point. Runs both DSPy modules (or heuristic fallback).

    Uses dspy.context(lm=lm) instead of dspy.configure() so that each
    Streamlit thread gets its own isolated DSPy settings -- this is the
    fix for:
      RuntimeError: dspy.settings can only be changed by the thread
      that initially configured it.
    """
    import dspy

    clear_debug_trace()

    if not transcript or len(transcript.strip()) < 10:
        reason = (
            f"Transcript too short ({len((transcript or '').strip())} chars). "
            "DSPy analysis skipped -- check Whisper output."
        )
        _trace("TRANSCRIPT_CHECK", "error", reason)
        log_action("DSPY_ANALYSIS", "Transcript too short -- skipping DSPy analysis.")
        return _heuristic_severity(transcript or "", fallback_reason=reason)

    _trace("TRANSCRIPT_CHECK", "ok", f"Transcript length OK ({len(transcript)} chars)")

    visual_obs = (
        vision_analysis.get("visual_flags", "No visual data.")
        + " "
        + vision_analysis.get("posture_assessment", "")
    ).strip()

    lm, build_reason = _build_lm()

    if lm is None:
        log_action("DSPY_ANALYSIS", f"Demo mode -- {build_reason}")
        return _heuristic_severity(transcript, fallback_reason=build_reason)

    # All DSPy calls inside dspy.context() -- thread-local, no global mutation
    try:
        with dspy.context(lm=lm):

            try:
                severity_mod = _get_severity_module()
                _trace("SEVERITY_MODULE_INIT", "ok", "ChainOfThought(SeveritySignature) instantiated")
            except Exception as e:
                reason = f"Failed to instantiate ChainOfThought severity module: {type(e).__name__}: {e}"
                _trace("SEVERITY_MODULE_INIT", "error", reason, exc=e)
                log_action("ERROR", reason, status="FAILED")
                return _heuristic_severity(transcript, fallback_reason=reason)

            try:
                sev_result = severity_mod(
                    transcript=transcript[:1500],
                    visual_observations=visual_obs[:500],
                )
                _trace("SEVERITY_MODULE_CALL", "ok",
                       f"Severity result: {sev_result.severity!r} | priority: {sev_result.triage_priority!r}")
            except Exception as e:
                reason = (
                    f"ChainOfThought severity call failed: {type(e).__name__}: {e}. "
                    "Possible causes: Mistral API rate limit, auth error, or malformed response from dspy."
                )
                _trace("SEVERITY_MODULE_CALL", "error", reason, exc=e)
                log_action("ERROR", reason, status="FAILED")
                return _heuristic_severity(transcript, fallback_reason=reason)

            try:
                risk_mod = _get_risk_module()
                _trace("RISK_MODULE_INIT", "ok", "Predict(RiskSignature) instantiated")
            except Exception as e:
                reason = f"Failed to instantiate Predict risk module: {type(e).__name__}: {e}"
                _trace("RISK_MODULE_INIT", "error", reason, exc=e)
                log_action("ERROR", reason, status="FAILED")
                return _heuristic_severity(transcript, fallback_reason=reason)

            try:
                risk_result = risk_mod(transcript=transcript[:1500])
                _trace("RISK_MODULE_CALL", "ok",
                       f"Risk flags: {risk_result.risk_flags!r} | urgent: {risk_result.urgent!r}")
            except Exception as e:
                reason = (
                    f"Predict risk-flag call failed: {type(e).__name__}: {e}. "
                    "Possible causes: Mistral API error or dspy output parsing failure."
                )
                _trace("RISK_MODULE_CALL", "error", reason, exc=e)
                log_action("ERROR", reason, status="FAILED")
                return _heuristic_severity(transcript, fallback_reason=reason)

    except Exception as e:
        reason = f"dspy.context() block failed unexpectedly: {type(e).__name__}: {e}"
        _trace("DSPY_CONTEXT", "error", reason, exc=e)
        log_action("ERROR", reason, status="FAILED")
        return _heuristic_severity(transcript, fallback_reason=reason)

    raw_flags = risk_result.risk_flags or "none"
    flags_list = (
        [f.strip() for f in raw_flags.split(",") if f.strip().lower() != "none"]
        if raw_flags.lower() != "none"
        else []
    )

    output = {
        "severity":              sev_result.severity.strip().lower(),
        "reasoning":             sev_result.reasoning.strip(),
        "triage_priority":       sev_result.triage_priority.strip(),
        "risk_flags":            flags_list,
        "urgent":                risk_result.urgent.strip().lower() == "yes",
        "confidence":            risk_result.confidence.strip().lower(),
        "dspy_active":           True,
        "debug_fallback_reason": "",
        "debug_trace":           get_debug_trace(),
    }

    _trace("PIPELINE_COMPLETE", "ok",
           f"DSPy pipeline done. severity={output['severity']}, urgent={output['urgent']}, "
           f"flags={len(flags_list)}")
    output["debug_trace"] = get_debug_trace()

    log_action(
        "DSPY_ANALYSIS",
        f"DSPy ChainOfThought complete. Severity={output['severity']}, "
        f"Urgent={output['urgent']}, Flags={len(flags_list)}.",
    )
    return output


def format_severity_badge(severity: str) -> tuple[str, str]:
    mapping = {
        "low":      ("🟢", "#2d6a4f"),
        "medium":   ("🟡", "#b5830a"),
        "high":     ("🔴", "#9b2226"),
        "critical": ("🚨", "#6a0572"),
    }
    return mapping.get(severity.lower(), ("⚪", "#555555"))