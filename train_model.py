import cv2
from cvzone.PoseModule import PoseDetector
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np


def get_label_from_key(key: int) -> str | None:
    mapping = {
        ord('u'): 'upright',
        ord('s'): 'slouched',
        ord('l'): 'leaning',
    }
    return mapping.get(key)


def collect_data() -> tuple[list[list[float]], list[str]]:
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    data: list[list[float]] = []
    labels: list[str] = []

    print("\nStarting data collection.")
    print("Press 'u' for upright, 's' for slouched, 'l' for leaning.")
    print("Press 'q' to stop collecting and train the model.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam. Exiting data collection.")
            break

        # Mirror the frame for a natural user experience
        frame = cv2.flip(frame, 1)

        # Detect the pose and draw the landmarks on the frame
        frame = detector.findPose(frame)
        lmList, _ = detector.findPosition(frame, draw=True)

        # Show the frame to the user
        cv2.imshow('Data Collection', frame)

        # Read a key; lower eight bits only
        key = cv2.waitKey(10) & 0xFF

        # If the user wants to quit, break the loop
        if key == ord('q'):
            break

        # Only record data if a pose was detected and a valid label key was pressed
        label = get_label_from_key(key)
        if lmList and label:
            # Flatten the list of (id, x, y) tuples into a simple list of floats
            row: list[float] = []
            for _, x, y in lmList:
                row.extend([float(x), float(y)])

            data.append(row)
            labels.append(label)

    cap.release()
    cv2.destroyAllWindows()

    return data, labels


def train_classifier(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    return clf


def evaluate_model(clf: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = clf.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")


def main() -> None:
    """Collect training data, train the classifier, and persist outputs."""
    data, labels = collect_data()

    if not data:
        print("No labelled data was collected. Exiting without training.")
        return

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    df['label'] = labels

    # Save raw labelled data for reproducibility
    df.to_csv('posture_data.csv', index=False)
    print(f"Collected {len(df)} labelled samples.")

    X = df.drop(columns='label')
    y = df['label']

    # Train the model
    classifier = train_classifier(X, y)

    # Persist the model and feature columns
    joblib.dump(classifier, 'posture_model.pkl')
    joblib.dump(list(X.columns), 'feature_columns.pkl')
    print("Saved trained model to 'posture_model.pkl' and feature columns to 'feature_columns.pkl'.")

    # Evaluate the model
    evaluate_model(classifier, X, y)


if __name__ == '__main__':
    main()
