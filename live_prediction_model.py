import cv2
from cvzone.PoseModule import PoseDetector
import joblib
import numpy as np


def main() -> None:
    """Run real‑time posture prediction on webcam input."""
    # Load the trained classifier and column order
    try:
        classifier = joblib.load('posture_model.pkl')
        columns = joblib.load('feature_columns.pkl')
    except FileNotFoundError:
        print("Model or feature column files are missing. Please run train_model.py first.")
        return

    detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    print("Starting live posture prediction. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to access the webcam. Exiting.")
            break

        # Mirror the frame so motion appears natural
        frame = cv2.flip(frame, 1)

        # Detect pose and get landmark list
        frame = detector.findPose(frame)
        lmList, _ = detector.findPosition(frame, draw=True)

        if lmList:
            # Flatten landmarks into row vector
            row: list[float] = []
            for _, x, y in lmList:
                row.extend([float(x), float(y)])
            # Predict only if the row length matches the training column count
            if len(row) == len(columns):
                X = np.array(row).reshape(1, -1)
                prediction = classifier.predict(X)[0]
                # Overlay the predicted label on the frame
                cv2.putText(
                    frame,
                    f"Posture: {prediction}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )

        # Display the frame to the user
        cv2.imshow('Live Posture Prediction', frame)
        # Exit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
