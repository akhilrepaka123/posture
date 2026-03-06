import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from pose_utils import build_detector, extract_landmarks, extract_posture_features

REMINDER_OVERLAY_DURATION = 3  # seconds to show "Correct your posture!" overlay


def _play_reminder_sound() -> None:
    """Play a system sound for posture reminder (cross-platform)."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif sys.platform == "win32":
            import winsound

            winsound.Beep(1000, 500)
        else:
            sound = "/usr/share/sounds/freedesktop/stereo/bell.oga"
            if os.path.exists(sound):
                subprocess.Popen(
                    ["paplay", sound],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
    except Exception:
        pass


def main() -> None:
    """Run real‑time posture prediction on webcam input."""
    # Load the trained classifier and column order
    try:
        classifier = joblib.load('posture_model.pkl')
        columns = joblib.load('feature_columns.pkl')
    except FileNotFoundError:
        print("Model or feature column files are missing. Please run train_model.py first.")
        return

    parser = argparse.ArgumentParser(description="Live posture prediction with optional reminders")
    parser.add_argument(
        "--reminder",
        type=float,
        default=0,
        metavar="SEC",
        help="Posture reminder threshold in seconds (0=disabled, e.g. 30)",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Log posture history to posture_history.csv",
    )
    args = parser.parse_args()
    reminder_threshold = args.reminder if args.reminder > 0 else None

    detector = build_detector()
    cap = cv2.VideoCapture(0)

    reminder_msg = f" (reminders every {reminder_threshold}s)" if reminder_threshold else ""
    print(f"Starting live posture prediction. Press 'q' to exit.{reminder_msg}")

    bad_posture_start: float | None = None
    reminder_overlay_until: float = 0
    posture_seconds: dict[str, float] = {"upright": 0.0, "slouched": 0.0, "leaning": 0.0}
    last_log_time: float | None = None
    log_interval = 2.0  # seconds between history log entries
    last_log_write: float = 0
    upright_streak_start: float | None = None
    current_streak_sec: float = 0
    best_streak_sec: float = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to access the webcam. Exiting.")
            break

        # Mirror the frame so motion appears natural
        frame = cv2.flip(frame, 1)

        # Detect pose and get landmark list
        frame, lmList = extract_landmarks(frame, detector)

        row = extract_posture_features(lmList) if lmList else None
        if row is not None and len(row) == len(columns):
            X = np.array(row).reshape(1, -1)
            proba = classifier.predict_proba(X)[0]
            classes = classifier.classes_
            pred_idx = np.argmax(proba)
            prediction = classes[pred_idx]
            confidence = float(proba[pred_idx])
            pct = int(round(confidence * 100))
            # Color: green >0.8, yellow 0.5-0.8, red <0.5
            if confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif confidence >= 0.5:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            streak_mins = int(current_streak_sec // 60)
            streak_secs = int(current_streak_sec % 60)
            cv2.putText(
                frame,
                f"Posture: {prediction} ({pct}%)",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Upright streak: {streak_mins}m {streak_secs}s",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Session summary: accumulate time per posture
            now = time.time()
            if last_log_time is not None:
                elapsed = now - last_log_time
                if prediction in posture_seconds:
                    posture_seconds[prediction] += elapsed
            last_log_time = now

            # Posture history logging
            if args.log and now - last_log_write >= log_interval:
                history_path = Path("posture_history.csv")
                write_header = not history_path.exists()
                try:
                    with open(history_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(["timestamp", "posture", "confidence"])
                        writer.writerow(
                            [
                                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                                prediction,
                                f"{confidence:.2f}",
                            ]
                        )
                    last_log_write = now
                except OSError:
                    pass

            # Upright streak
            if prediction == "upright":
                if upright_streak_start is None:
                    upright_streak_start = now
                current_streak_sec = now - upright_streak_start
                if current_streak_sec > best_streak_sec:
                    best_streak_sec = current_streak_sec
            else:
                upright_streak_start = None
                current_streak_sec = 0

            # Posture reminder: track time in bad posture
            is_bad = prediction in ("slouched", "leaning")
            now = time.time()
            if is_bad and reminder_threshold:
                if bad_posture_start is None:
                    bad_posture_start = now
                elif now - bad_posture_start >= reminder_threshold:
                    _play_reminder_sound()
                    bad_posture_start = now
                    reminder_overlay_until = now + REMINDER_OVERLAY_DURATION
            else:
                bad_posture_start = None

        else:
            last_log_time = None
            upright_streak_start = None
            cv2.putText(
                frame,
                "Position yourself in frame",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 165, 255),  # Orange
                2,
            )

        if time.time() < reminder_overlay_until:
            cv2.putText(
                frame,
                "Correct your posture!",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )

        # Display the frame to the user
        cv2.imshow('Live Posture Prediction', frame)
        # Exit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Session summary
    total = sum(posture_seconds.values())
    if total > 0:
        mins = int(total // 60)
        secs = int(total % 60)
        print(f"\n--- Session Summary ---")
        print(f"Total time: {mins}m {secs}s")
        for posture, sec in posture_seconds.items():
            pct = 100 * sec / total
            print(f"  {posture}: {pct:.0f}%")
        slouch_mins = int(posture_seconds["slouched"] // 60)
        slouch_secs = int(posture_seconds["slouched"] % 60)
        print(f"You slouched ~{slouch_mins}m {slouch_secs}s")
        best_mins = int(best_streak_sec // 60)
        best_secs = int(best_streak_sec % 60)
        print(f"Best upright streak: {best_mins}m {best_secs}s")


if __name__ == '__main__':
    main()