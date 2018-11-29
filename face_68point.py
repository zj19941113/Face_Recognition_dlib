# coding=utf-8

import cv2
import dlib

path = "test-face/0001_IR_angry.jpg"
img = cv2.imread(path)

# 人脸分类器
detector = dlib.get_frontal_face_detector()
# 获取人脸检测器
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

dets = detector(img, 1)
for face in dets:
    shape = predictor(img, face)  # 寻找人脸的68个标定点
    # 遍历所有点，打印出其坐标，并圈出来
    for pt in shape.parts():
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
    cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()