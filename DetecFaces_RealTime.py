import cv2

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

##video capture
video_capture = cv2.VideoCapture(0)

##setting for video resolution
video_capture.set(3, 640)
video_capture.set(4, 480)

while True:
    #returns the image frames
    ret, img = video_capture.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_faces = cascade_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=8, minSize=(20,20))

    for (x,y, width, height) in detected_faces:
         cv2.rectangle(img, (x,y), (x+width, y+height), (0,0,2550,20))

    cv2.imshow('RealTime Face Detection', img)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()