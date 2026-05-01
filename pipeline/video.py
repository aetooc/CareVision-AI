import cv2
import os
import numpy as np
from pipeline.logging_utils import log_action

# Optional: scikit-learn for lightweight anomaly detection on frame features
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def _extract_frame_features(frame: np.ndarray) -> dict:
    """
    Computes lightweight visual features from a single BGR frame.
    Returns a dict suitable for downstream aggregation / ML scoring.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Brightness & contrast
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))

    # Face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_count = len(faces)

    # Motion proxy: Laplacian variance (blur ↔ movement)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Saturation (low → grayscale / poor lighting)
    saturation = float(np.mean(hsv[:, :, 1]))

    return {
        "brightness": brightness,
        "contrast": contrast,
        "face_count": face_count,
        "laplacian_var": laplacian_var,
        "saturation": saturation,
    }


def _run_anomaly_detection(features_list: list[dict]) -> list[int]:
    """
    Runs IsolationForest on per-frame features to flag anomalous frames
    (e.g., sudden darkness, patient leaving frame).
    Returns a list of frame indices marked as anomalous (-1).
    """
    if not _SKLEARN_AVAILABLE or len(features_list) < 4:
        return []

    X = np.array([
        [f["brightness"], f["contrast"], f["face_count"], f["laplacian_var"]]
        for f in features_list
    ])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = IsolationForest(contamination=0.1, random_state=42)
    labels = clf.fit_predict(X_scaled)          # -1 = anomaly, 1 = normal
    return [i for i, l in enumerate(labels) if l == -1]


def process_video_frames(video_path: str, frames_dir: str) -> dict:
    """
    Extracts 1 frame per second, computes per-frame OpenCV features,
    and runs an IsolationForest anomaly scorer on the feature matrix.

    Returns a vision_analysis dict consumed by the LangChain agent.
    """
    cap = cv2.VideoCapture(video_path)
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = round(total_frames / fps, 1) if fps else 0

    # Clear stale frames
    for f in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, f))

    frame_count = 0
    saved_count = 0
    features_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:
            frame_path = os.path.join(frames_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            features_list.append(_extract_frame_features(frame))
            saved_count += 1

        frame_count += 1

    cap.release()
    log_action("DATA_EXTRACTION", f"Extracted {saved_count} frames at 1 FPS (video: {duration_s}s).")

    # ── Aggregate features ─────────────────────────────────────────────────────
    if not features_list:
        return {"frames_analyzed": 0, "visual_flags": "No frames extracted.", "posture_assessment": "N/A"}

    avg_brightness  = np.mean([f["brightness"]    for f in features_list])
    avg_contrast    = np.mean([f["contrast"]       for f in features_list])
    avg_faces       = np.mean([f["face_count"]     for f in features_list])
    avg_saturation  = np.mean([f["saturation"]     for f in features_list])
    anomalous_frames = _run_anomaly_detection(features_list)

    # ── Build human-readable flags ─────────────────────────────────────────────
    flags = []
    if avg_faces >= 0.7:
        flags.append("Patient consistently visible in frame.")
    else:
        flags.append("Patient intermittently out of frame — check positioning.")

    if avg_brightness < 60:
        flags.append("Lighting suboptimal (dark environment).")
    elif avg_brightness > 200:
        flags.append("Overexposed — possible window glare.")
    else:
        flags.append("Lighting adequate.")

    if anomalous_frames:
        flags.append(f"Anomalous frames detected at seconds: {anomalous_frames[:5]}.")

    # Simple posture proxy: sustained face presence + moderate motion variance
    motion_var = np.std([f["laplacian_var"] for f in features_list])
    if avg_faces >= 0.8 and motion_var < 500:
        posture = "Patient appears seated and stable throughout consultation."
    elif avg_faces >= 0.5:
        posture = "Patient visible with moderate movement; possible discomfort or repositioning."
    else:
        posture = "Patient position unclear — insufficient face detections."

    log_action(
        "AI_PROCESSING",
        f"Frame analysis complete. Avg brightness={avg_brightness:.1f}, "
        f"faces={avg_faces:.2f}, anomalies={len(anomalous_frames)}.",
    )

    return {
        "frames_analyzed": saved_count,
        "duration_seconds": duration_s,
        "avg_brightness": round(float(avg_brightness), 1),
        "avg_contrast": round(float(avg_contrast), 1),
        "avg_face_presence": round(float(avg_faces), 2),
        "anomalous_frame_count": len(anomalous_frames),
        "visual_flags": " ".join(flags),
        "posture_assessment": posture,
        "sklearn_anomaly_detection": _SKLEARN_AVAILABLE,
    }
