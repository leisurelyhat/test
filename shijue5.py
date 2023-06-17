
import cv2
import numpy as np
import dlib
import torch
from mtcnn.mtcnn import MTCNN
from facenet_pytorch import MTCNN as FaceNet_MTCNN, InceptionResnetV1, prewhiten
from scipy.spatial.distance import euclidean




# 创建MTCNN对象用于人脸检测
detector = MTCNN()
# 加载Dlib预训练模型，用于从人脸中提取68个关键点
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# 创建FaceNet模型和MTCNN对象用于处理人脸
facenet_mtcnn = FaceNet_MTCNN(keep_all=True)
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# 预计算数据库中的人脸特征向量（这里应更改为实际数据库图片路径）
db_images = ['c1.jpg', 'c2.jpg', 'c3.jpg']
face_features = []
for img_name in db_images:
    img = cv2.imread(img_name)
    aligned_face = facenet_mtcnn(img)
    aligned_faces = facenet_mtcnn(img)
    if len(aligned_faces) > 0:
        aligned_face = aligned_faces[0]
        face_feature = facenet_model(aligned_face.unsqueeze(0)).detach().numpy()
        face_features.append(face_feature)
    else:
        continue

    

# 初始化5个预设参考点，用于人脸对齐
REFERENCE_POINTS = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]], dtype=np.float32)

def align_face(img, face_keypoints, output_size=(96, 112)):
    confidence_points = np.zeros((5, 2), np.float32)
    confidence_points[0] = face_keypoints['left_eye']
    confidence_points[1] = face_keypoints['right_eye']
    confidence_points[2] = face_keypoints['nose']
    confidence_points[3] = face_keypoints['mouth_left']
    confidence_points[4] = face_keypoints['mouth_right']

    T = cv2.estimateAffinePartial2D(confidence_points, REFERENCE_POINTS)[0]
    aligned = cv2.warpAffine(img, T, output_size)
    return aligned

def draw_68_keypoints(img, result):
    for i in range(1, 68):
        x = result.part(i).x
        y = result.part(i).y
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

def match_faces(query_face_feature, threshold=8.0):
    distances = [euclidean(query_face_feature, feature) for feature in face_features]
    closest_index, closest_distance = min(enumerate(distances), key=lambda x: x[1])

    if closest_distance < threshold:
        return closest_index, closest_distance
    else:
        return None, closest_distance

def detect_faces_and_recognize(filename):
    pixels = cv2.imread(filename)
    faces = detector.detect_faces(pixels)
    face_rois = []

    for result in faces:
        x, y, width, height = result['box']
        cv2.rectangle(pixels, (x, y), (x + width, y + height), (255, 0, 0), 2)
        face_roi = pixels[y:y + height, x:x + width]
        aligned_face = align_face(face_roi, result['keypoints'])
        face_rois.append(aligned_face)

    test_input = torch.stack([torch.from_numpy(prewhiten(face)) for face in face_rois])
    test_output = facenet_model(test_input).detach().numpy()
    
    for index, face_feature in enumerate(test_output):
        face_index, distance = match_faces(face_feature)
        if face_index is not None:
            matched_face_name = db_images[face_index].split('.')[0]
            x, y, width, height = faces[index]['box']
            cv2.putText(pixels, f"{matched_face_name} ({distance:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        else:
            x, y, width, height = faces[index]['box']
            cv2.putText(pixels, f"Unknown ({distance:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow('faces_detected', pixels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

filename = "3.png"  
detect_faces_and_recognize(filename)
