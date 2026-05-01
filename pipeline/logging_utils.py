import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_LOG_FILE = os.path.join(BASE_DIR, "logs", "audit_log.json")


def log_action(action_type: str, details: str, status: str = "SUCCESS") -> dict:
    """
    Appends a timestamped entry to the JSON audit log.
    Thread-safe for single-process Streamlit usage.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action_type": action_type,
        "status": status,
        "details": details,
    }

    os.makedirs(os.path.dirname(AUDIT_LOG_FILE), exist_ok=True)

    try:
        with open(AUDIT_LOG_FILE, "r") as f:
            logs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logs = []

    logs.append(entry)

    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

    return entry
