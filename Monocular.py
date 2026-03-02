import cv2

# stop_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:

    if not ret:
        break

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = stop_cascade.detectMultiScale(img_gray, minSize=(20, 20))

    for (x, y, w, h) in found:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()