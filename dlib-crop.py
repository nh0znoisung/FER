# https://medium0.com/m/global-identity?redirectUrl=https%3A%2F%2Ftowardsdatascience.com%2Fface-landmark-detection-using-python-1964cb620837

import cv2
import dlib
import numpy as np

# Detector one or more face
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
detector = dlib.get_frontal_face_detector()
# Detect key point from face. Params is pre-trained model
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# cap = cv2.VideoCapture(0)

# Download pre-trained model: https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat
# The model is based on ensemble regression tree because the model will predict continuous number
# Paper (2014): http://kth.diva-portal.org/smash/get/diva2:713097/FULLTEXT01
# Dataset: iBUG-300 W dataset (Image and 68 landmarks of face)
# Link dataset: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
# 

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

OFFSET_X = 10
OFFSET_Y = 10

ATTR = [17,18,19,20,21,22,23,24,25,26,36,37,38,39,40,41,42,43,44,45,46,47]

def get_edge(landmarks):
    x_min,x_max,y_min,y_max = 1e9,-1e9,1e9,-1e9
    for attr in ATTR:
        if not landmarks.part(attr):
            return -1e9,0,0,0
        x_min = min(x_min,landmarks.part(attr).x)
        x_max = max(x_max,landmarks.part(attr).x)
        y_min = min(y_min,landmarks.part(attr).y)
        y_max = max(y_max,landmarks.part(attr).y)

    return x_min, y_min, x_max, y_max

# Crop only eyes and eyebrows
def crop_image(image, is_draw=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect the face
    rects = detector(gray, 1)
    if(len(rects) == 0):
        return None
    # Detect landmarks for each face
    for rect in rects:
        # Get the landmark points
        landmarks = predictor(gray, rect)
        if is_draw:
            tmp_image = image.copy()
            for attr in ATTR:
                cv2.circle(tmp_image, (landmarks.part(attr).x, landmarks.part(attr).y), 1, (0, 0, 255), -1)
                cv2.imwrite("image_draw.png", tmp_image)
        x_min, y_min, x_max, y_max = get_edge(landmarks)
        if x_min == -1e9:
            return None
        top = y_min - OFFSET_Y
        bottom = y_max + OFFSET_Y
        left = x_min - OFFSET_X
        right = x_max + OFFSET_X
        try:
            cropped = image[top:bottom, left:right]
            cropped = cv2.resize(cropped, (260, 110), interpolation=cv2.INTER_LINEAR)

            norm = np.zeros((260, 110))
            cropped = cv2.normalize(cropped, norm, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite("image.png", cropped)
            return cropped
        except:
            return None

# Crop whole face
def crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detectMultiScale(
    #         gray,
    #         scaleFactor=1.1,
    #         minNeighbors=7,
    #         minSize=(100, 100),)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if(len(faces) == 0):
        return None
    (x,y,w,h) = faces[0]
    cropped = image[y:y+h, x:x+w]
    cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)

    norm = np.zeros((224,224))
    cropped = cv2.normalize(cropped, norm, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("image.png", cropped)
    return cropped

# ImageDataGenerator(
# featurewise_center=False,
# featurewise_std_normalization=False,
# rotation_range=10,
# width_shift_range=0.1,
# height_shift_range=0.1,
# zoom_range=.1,
# horizontal_flip=True)

import os

def process(file_path:str):
    image = cv2.imread(file_path)
    # Use crop_image() for crop eyes and eyebrows
    # Use crop_faec() for face detection
    cropped_image = crop_image(image, True)
    if cropped_image is None:
        return
    target_path = file_path.replace("/M-LFW-FER/", "/M-LFW-FER-face-detect/")
    dir_path = os.path.dirname(target_path)
    os.makedirs(dir_path, exist_ok=True)
    try:
        cv2.imwrite(target_path, cropped_image)
    except:
        return
    


process("datasets/M-LFW-FER/eval/positive/Sue_Slavec_0001.jpg")

