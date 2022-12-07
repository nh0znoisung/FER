# https://medium0.com/m/global-identity?redirectUrl=https%3A%2F%2Ftowardsdatascience.com%2Fface-landmark-detection-using-python-1964cb620837
# Github: https://gist.github.com/khalidmeister/4667ba516aa96eea250032c26806e2af

# 

# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYEBROW

# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE
# from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYEBROW



# face_mest_test.py
# IMPORTING LIBRARIES
import cv2
import mediapipe as mp
import numpy as np


_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5
_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
BLUE_COLOR = (255, 0, 0)

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh


connections_list = {
	"left_eye": mp_face_mesh.FACEMESH_LEFT_EYE,
	"left_eyebrow": mp_face_mesh.FACEMESH_LEFT_EYEBROW,
	"right_eye": mp_face_mesh.FACEMESH_RIGHT_EYE,
	"right_eyebrow": mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
}

def connections_to_points(connections):
	points = set()
	for connection in connections:
		start_idx = connection[0]
		end_idx = connection[1]
		points.add(start_idx)
		points.add(end_idx)
	return list(points)

points_list = {k:connections_to_points(v)   for (k,v) in connections_list.items()}
points_all = []
for value in points_list.values():
	points_all += value
# print(len(points_all))
points_list["all"] = points_all

# for points in points_list:
# 	print(len(points))

# 16: eye > 6
# 10: eye_brow > 6
# 52 points

# print(connections_to_points(connections_list[0]))
# print(connections_list[0])
# Call in landmark
def get_idx_to_coordinates(landmark_list, image):
	image_rows, image_cols, _ = image.shape
	idx_to_coordinates = {}
	for idx, landmark in enumerate(landmark_list.landmark):
		if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
			continue
		landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
													image_cols, image_rows)
		if landmark_px:
			idx_to_coordinates[idx] = landmark_px
	return idx_to_coordinates

def points_to_pixel_coordinates(points, idx_to_coordinates):
	pc_list = []
	# print(points)
	# print(idx_to_coordinates)
	for point in points:
		if point in idx_to_coordinates:
			pc_list.append(idx_to_coordinates[point])
	return pc_list
		

# refine_landmarks = True => refine the landmark coordinates around the eyes and lips
# MAX_NUM_FACES = 1

# DETECT THE FACE LANDMARKS
face_mesh = mp_face_mesh.FaceMesh(
				min_detection_confidence=0.5, 
				min_tracking_confidence=0.5,
				refine_landmarks=True
			)

def process(file_path):
	image = cv2.imread(file_path)	
	
	image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
	image.flags.writeable = False

	results = face_mesh.process(image)

	image.flags.writeable = True
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


	# Draw the face mesh annotations on the image.
	if not results.multi_face_landmarks:
		return None

	# only 1 face
	landmark_list = results.multi_face_landmarks[0]
	idx_to_coordinates = get_idx_to_coordinates(landmark_list, image)
	return idx_to_coordinates

# # connections = connections_list[0]
# points = points_list["left_eyebrow"]
# 
# # print(idx_to_coordinates)
# pixel_coordinates = points_to_pixel_coordinates(points, idx_to_coordinates)

# annotated_image = image.copy()
# draw_image(annotated_image, pixel_coordinates)
# write_image(annotated_image, dest_path)

SPEC = ["left_eyebrow","left_eye","right_eyebrow","right_eye","all"]
# 250*250
def get_coordinates(file_path, spec="all"):
	idx_to_coordinates = process(file_path)
	if not idx_to_coordinates:
		return None
	if spec not in SPEC:
		return None
	points = points_list[spec] #list(int)
	pixel_coordinates = points_to_pixel_coordinates(points, idx_to_coordinates)
	# list(tuple(int))
	return np.array(pixel_coordinates) #2d-array


# draw_image(annotated_image, pixel_coordinates)
# write_image(annotated_image, dest_path)

def draw_image(annotated_image, pixel_coordinates):
	for pixel_coordinate in pixel_coordinates:
		cv2.circle(annotated_image, pixel_coordinate, 1, (0, 0, 255), -1)

def write_image(image, dest_path):
	cv2.imwrite(dest_path, image)

# datasets/M-LFW-FER/train/positive/George_HW_Bush_0010.jpg
# abcd.png
# print(process("datasets/M-LFW-FER/train/positive/George_HW_Bush_0010.jpg", "abc.png"))
# X = "datasets/M-LFW-FER/train/positive/George_HW_Bush_0010.jpg"
# get_coordinates(X,"left_eyebrow")

# cap = cv2.VideoCapture(0)
# whil True
# success, image = cap.read()
	# # Terminate the process
	# if cv2.waitKey(5) & 0xFF == 27:
	# 	break
# cap.release()

# glob get path

def main(): pass

X = [(153, 111), (139, 108), (140, 105), (157, 116), (147, 107), (133, 104), (154, 116), (146, 109), (150, 112), (131, 108)]
X = np.array(X)
def normalize_2d(matrix):
	temp_mat = np.zeros(matrix.shape)
	_, col = matrix.shape
	for i in range(col):
		temp_mat[:, i] = normalize_1d(matrix[:, i])
	return temp_mat

def normalize_1d(lst):
	mn = min(lst)
	mx = max(lst)
	temp = np.array(list(map(lambda x: (x-mn)/(mx-mn), lst)))
	# print(temp)
	return temp

# print(normalize_2d(X))
# print(X.flatten())

# a = get_coordinates("datasets/M-LFW-FER/train/positive/Roger_Grimes_0001.jpg")
# b = get_coordinates("datasets/M-LFW-FER-face-cut/train/positive/Roger_Grimes_0001.jpg","left_eye")

# print(a)
# print(b)

import glob, os
# os.chdir("datasets/M-LFW-FER/train/positive")
# for file in glob.glob("*.jpg"):
#     print(file)

EMOTION = ["positive", "negative", "neutral"]
DATASETS = ["LFW-FER","M-LFW-FER","M-LFW-FER-face-cut"]
TRAIN = ["train", "eval"]

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def parse_datasets(path):
	for dataset in DATASETS:
		datasets_path = path + "/" + dataset
		print(f"Run in {datasets_path}")
		for train in TRAIN:
			train_path = datasets_path + "/" + train
			print(f"*Run in {train_path}")
			lst = []
			# f = open(f"{datasets_path}/{train}.csv", "a")
			for idx,emotion in enumerate(EMOTION):
				emotion_path = train_path + "/" + emotion
				print(f"**Run in {emotion_path}")
				for file_path in glob.glob(emotion_path + "/*.jpg"):
					# print(file_path)
					coordinates = get_coordinates(file_path)
					if coordinates is None:
						continue
					coordinates_normalized = normalize_2d(coordinates)
					coordinates_final = coordinates_normalized.flatten()
					coordinates_final = np.append(coordinates_final, idx).astype("float64")
					lst.append(coordinates_final)
					# f.write(",".join(tmp))
			
			# f.close()
			np.savetxt(f"{datasets_path}/{train}.csv", lst, delimiter=",")
			print(f"**Save successfully**")



# print(parse_emotion("datasets/M-LFW-FER/train/positive"))

parse_datasets("datasets")
