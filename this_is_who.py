# -*- coding: UTF-8 -*-

import os, dlib, numpy
import cv2

# 1.人脸关键点检测器
predictor_path = "shape_predictor_68_face_landmarks.dat"

# 2.人脸识别模型
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

# 3.候选人文件
candidate_npydata_path = "candidates.npy"
candidate_path = "candidates.txt"

# 4.需识别的人脸文件夹
img_dir = "test-face"

# 5.识别结果存放文件夹
faceRect_path = "faceRec"

# 6.识别错误结果存放文件夹
faceRect_ERROR_path = "faceRec_ERROR"

# 7.相似度阈值，高于此值为非人脸库数据，显示unknow
is_not_candidate = 0.6


# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()

# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)

# 3. 加载人脸识别模型
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

filelist = os.listdir(img_dir)
count = right_num = 0
for file in filelist:
    img_path = os.path.join(img_dir, file)
    # 提取描述子
    print("Processing file: {}".format(img_path))

    img = cv2.imread(img_path)
    dets = detector(img, 1)
    # print("Number of faces detected: {}".format(len(dets)))

    for k, d in enumerate(dets):
        shape = sp(img, d)
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)
            cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        d_test2 = numpy.array(face_descriptor)
        # 计算欧式距离
        dist = []
        for i in descriptors:
            dist_ = numpy.linalg.norm(i - d_test2)
            dist.append(dist_)
        if (min(dist)) > is_not_candidate:
            this_is = "Unknow"
        else:
            num = dist.index(min(dist))  # 返回最小值
            this_is = candidate[num][0:4]
        # print( min(dist))

        left_top = (dlib.rectangle.left(d), dlib.rectangle.top(d))
        right_bottom = (dlib.rectangle.right(d), dlib.rectangle.bottom(d))
        cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 2, cv2.LINE_AA)
        text_point = (dlib.rectangle.left(d), dlib.rectangle.top(d) - 5)
        cv2.putText(img, this_is, text_point, cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2, 1)  # 标出face

    if this_is == file[0:4]:
       right_num += 1
    else:
        print("Processing file: ",img_path," ERROR !")
        cv2.imwrite(os.path.join(faceRect_ERROR_path, file+"_to_"+this_is+".jpg"), img)

    cv2.imwrite(os.path.join(faceRect_path,file), img)

    count += 1

accuracy = right_num/count

print("There are ",count," pictures")
print("Identify ",right_num," photos correctly, accuracy： ",accuracy)

