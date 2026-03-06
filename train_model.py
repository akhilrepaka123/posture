import cv2
import pandas as pd

from pose_utils import FEATURE_COLUMNS, build_detector, extract_landmarks, extract_posture_features
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
    detector = build_detector()
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
        frame, lmList = extract_landmarks(frame, detector)

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
            row = extract_posture_features(lmList)
            if row is not None:
                data.append(row)
                labels.append(label)

    cap.release()
    cv2.destroyAllWindows()

    return data, labels


def train_classifier(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    return clf


def evaluate_model(clf: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
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
    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    df['label'] = labels

    # Save raw labelled data for reproducibility
    df.to_csv('posture_data.csv', index=False)
    print(f"Collected {len(df)} labelled samples.")

    X = df.drop(columns='label')
    y = df['label']

    # Split before training to avoid data leakage
    stratify_arg = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )

    # Train on train set only
    classifier = train_classifier(X_train, y_train)

    # Persist the model and feature columns
    joblib.dump(classifier, 'posture_model.pkl')
    joblib.dump(FEATURE_COLUMNS, 'feature_columns.pkl')
    print("Saved trained model to 'posture_model.pkl' and feature columns to 'feature_columns.pkl'.")

    # Evaluate on held-out test set
    evaluate_model(classifier, X_test, y_test)


if __name__ == '__main__':
    main()