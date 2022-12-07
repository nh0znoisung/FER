# https://soumyapatilblogs.medium.com/face-and-eyes-detection-using-opencv-9fcad47656a4

# Dataset of OpenCV: https://github.com/opencv/opencv/tree/master/data/haarcascades
import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
#save the image(i) in the same directory
img = cv2.imread("friends.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
(x,y,w,h) = faces[0]
print(x,x+w,y,y+h)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cropped = img[y:y+h, x:x+w]
cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)

norm = np.zeros((224,224))
cropped = cv2.normalize(cropped, norm, 0, 255, cv2.NORM_MINMAX)

# rows, cols = cropped.shape[:2]
# M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
# cropped = cv2.warpAffine(cropped, M, (cols,rows))

cv2.imwrite("image.png", cropped)