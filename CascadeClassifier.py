import cv2
import matplotlib.pyplot as plt

cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

image = cv2.imread('image.jpg')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detected_faces = cascade_classifier.detectMultiScale(gray_image, 1.1, 1)
##for green square around the face
for (x, y, width, height) in detected_faces:
    cv2.rectangle(image,(x,y), (x+width, y+height), (0, 0, 255), 10)

##openCV uses BGR and ,atplotlib RGB
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

