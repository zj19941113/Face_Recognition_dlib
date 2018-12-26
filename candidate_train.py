# -*- coding: UTF-8 -*-

import sys,os,dlib,numpy
import cv2

# 1.人脸关键点检测器
predictor_path = "shape_predictor_68_face_landmarks.dat"

# 2.人脸识别模型
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 3.候选人脸文件夹
faces_folder_path = "candidate-face"

# 4.需识别的人脸
img_path = "test-face/0001_IR_allleft.jpg"

# 5.识别结果存放文件夹
faceRect_path = "faceRec"


# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()

# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)

# 3. 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)



# 候选人脸描述子list

candidates = []

filelist = os.listdir(faces_folder_path)
count = 0
for fn in filelist:
        count = count+1
descriptors = numpy.zeros(shape=(count,128))
n = 0
for file in filelist:
    f = os.path.join(faces_folder_path,file)
    #if os.path.splitext(file)[1] == ".jpg" #文件扩展名
    print("Processing file: {}".format(f))
    img = cv2.imread(f)
    # 1.人脸检测
    dets = detector(img, 1)

    for k, d in enumerate(dets):
        # 2.关键点检测
        shape = sp(img, d)

        # 3.描述子提取，128D向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        # 转换为numpy array
        v = numpy.array(face_descriptor)

        descriptors[n] = v

        # descriptors.append(v)
        candidates.append(os.path.splitext(file)[0])

    n += 1

    for d in dets:
        # print("faceRec locate:",d)
        # print(type(d))
        # 使用opencv在原图上画出人脸位置
        left_top = (dlib.rectangle.left(d), dlib.rectangle.top(d))
        right_bottom = (dlib.rectangle.right(d), dlib.rectangle.bottom(d))
        cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 2, cv2.LINE_AA)

    # cv2.imwrite(os.path.join(faceRect_path,file), img)

numpy.save('candidates.npy',descriptors)
file= open('candidates.txt', 'w')
for candidate in candidates:
    file.write(candidate)
    file.write('\n')
file.close()



