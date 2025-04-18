import cv2
from cvzone.PoseModule import PoseDetector
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

detector = PoseDetector()
cap = cv2.VideoCapture(0)

data = []
labels = []

def get_label_from_key(key):
    return {
        ord('u'): 'upright',
        ord('s'): 'slouched',
        ord('l'): 'leaning'
    }.get(key, None)

print("Press 'u'=upright, 's'=slouched, 'l'=leaning, 'q'=quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, draw=True)

    if lmList:
        row = [val for _, x, y in lmList for val in (x, y)]

        cv2.imshow("Collecting Data", img)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        label = get_label_from_key(key)
        if label:
            data.append(row)
            labels.append(label)
    else:
        cv2.imshow("Collecting Data", img)
        cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()

# Save and train model
df = pd.DataFrame(data)
df['label'] = labels
X = df.drop(columns='label')
y = df['label']

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, y)

joblib.dump(clf, 'posture_model.pkl')
joblib.dump(list(X.columns), 'feature_columns.pkl')

# Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
preds = clf.predict(X_test)
print(classification_report(y_test, preds))
