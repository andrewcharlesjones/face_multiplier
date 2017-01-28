import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    for (x, y, w, h) in faces:
        if x + w < frame.shape[1]:
            if x-w > 0:
                frame[y:y+h, x-w:x] = frame[y:y+h, x:w+x]
            if x+2*w < frame.shape[1]:
                frame[y:y+h, x+w:x+2*w] = frame[y:y+h, x:x+w]

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()