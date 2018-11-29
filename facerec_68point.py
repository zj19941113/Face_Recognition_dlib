# -*- coding: UTF-8 -*-

import dlib
import cv2
import numpy

# 待检测图片
img_path = "all-face.jpg"
# 人脸关键点检测器
predictor_path="shape_predictor_68_face_landmarks.dat"
# 人脸识别模型
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
# 候选人文件
candidate_npydata_path = "candidates.npy"
candidate_path = "candidates.txt"

# 加载正脸检测器
detector = dlib.get_frontal_face_detector()
# 加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)
# 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)


# 候选人脸描述子list

# 读取候选人数据
npy_data=numpy.load(candidate_npydata_path)
descriptors=npy_data.tolist()

# 候选人名单
candidate = []
file=open(candidate_path, 'r')
list_read = file.readlines()
for name in list_read:
    name = name.strip('\n')
    candidate.append(name)

print("Processing file: {}".format(img_path))
img = cv2.imread(img_path)

# 1.人脸检测
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))



for k, d in enumerate(dets):
    # 2.关键点检测
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test2 = numpy.array(face_descriptor)
    # 计算欧式距离
    dist = []
    for i in descriptors:
        dist_ = numpy.linalg.norm(i - d_test2)
        dist.append(dist_)
    num = dist.index(min(dist))  # 返回最小值

    left_top = (dlib.rectangle.left(d), dlib.rectangle.top(d))
    right_bottom = (dlib.rectangle.right(d), dlib.rectangle.bottom(d))
    cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 2, cv2.LINE_AA)
    text_point = (dlib.rectangle.left(d), dlib.rectangle.top(d) - 5)
    cv2.putText(img, candidate[num], text_point, cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2, 1)  # 标出face

cv2.imwrite('all-face-result.jpg', img)

# cv2.imshow("img",img) # 转成ＢＧＲ显示
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()