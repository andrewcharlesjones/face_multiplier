import cv2
import sys
from scipy.misc import imresize
import select

if len(sys.argv) == 1:
    cascPath = 'haarcascade_frontalface_default.xml'
else:
    cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

size = 1

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
        face = frame[y:y+h, x:w+x]
        intw = int(w/2)
        inth = int(h/2)
        smaller_face = imresize(face, (inth, intw, 3))

        # cv2.rectangle(frame, (x, y), (x+w, y+h), 2)
        if x + w < frame.shape[1]:
            if x-w > 0:
                frame[y:y+h, x-w:x] = face
            if x+2*w < frame.shape[1]:
                frame[y:y+h, x+w:x+2*w] = face
                
                inc = intw
                if w - intw > inth:
                    # print 'switching'
                    inc = intw + 1
                # print 'small', inth, x - w
                # print smaller_face.shape
                if smaller_face.shape[1] != 0 and size == 0:
                    frame[y:y+intw, x-w:x-inc] = smaller_face
        if y + h < frame.shape[0]:
            if y - h > 0:
                frame[y-h:y, x:x+w] = face
            if y + 2 * h < frame.shape[0]:
                frame[y+h:y+2*h, x:x+w] = face


    cv2.imshow('Video', frame)

    # print intw, inth, len([y+inth]), len([x-w:x-intw-1])
    # print 'small', (y, y+inth), (x-w, x-intw)
    # print smaller_face.shape

    # if cv2.waitKey(1) & 0xFF == ord('a'):
    #     frame[y:y+inth, x-w:x-intw] = smaller_face
    #     print 'big'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('a'):
        size = 0


cap.release()
cv2.destroyAllWindows()
