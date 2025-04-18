import cv2
from cvzone.PoseModule import PoseDetector
import joblib

# Load trained model + column structure
model = joblib.load('posture_model.pkl')
columns = joblib.load('feature_columns.pkl')

detector = PoseDetector()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img = detector.findPose(img)
    lmList, _ = detector.findPosition(img, draw=True)

    if lmList:
        row = [val for _, x, y in lmList for val in (x, y)]
        if len(row) == len(columns):
            prediction = model.predict([row])[0]
            cv2.putText(img, f'Posture: {prediction}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Live Posture Prediction", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
