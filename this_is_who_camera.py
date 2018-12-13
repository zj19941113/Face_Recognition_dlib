# -*- coding: UTF-8 -*-

import dlib,numpy 
import cv2          
import time

# 1.人脸关键点检测器
predictor_path = "shape_predictor_68_face_landmarks.dat"
# 2.人脸识别模型
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
# 3.候选人文件
candidate_npydata_path = "candidates.npy"
candidate_path = "candidates.txt"
# 4.储存截图目录
path_screenshots = "screenShots/"


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

# 创建 cv2 摄像头对象
cv2.namedWindow("camera", 1)
cap = cv2.VideoCapture(0)
cap.set(3, 480)
# 截图 screenshots 的计数器
cnt = 0
while (cap.isOpened()):  #isOpened()  检测摄像头是否处于打开状态
    ret, img = cap.read()  #把摄像头获取的图像信息保存之img变量
    if ret == True:       #如果摄像头读取图像成功
        # 添加提示
        cv2.putText(img, "press 'S': screenshot", (20, 420), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "press 'Q': quit", (20, 440), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
        dets = detector(img, 1)
        if len(dets) != 0:
            # 检测到人脸
            for k, d in enumerate(dets):
                # 关键点检测
                shape = sp(img, d)
                # 遍历所有点圈出来
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
                num = dist.index(min(dist))  # 返回最小值

                left_top = (dlib.rectangle.left(d), dlib.rectangle.top(d))
                right_bottom = (dlib.rectangle.right(d), dlib.rectangle.bottom(d))
                cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), 2, cv2.LINE_AA)
                text_point = (dlib.rectangle.left(d), dlib.rectangle.top(d) - 5)
                cv2.putText(img, candidate[num][0:4], text_point, cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 1, 1)  # 标出face

            cv2.putText(img, "facesNum: " + str(len(dets)), (20, 50),  cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            # 没有检测到人脸
            cv2.putText(img, "facesNum:0", (20, 50),  cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

        k = cv2.waitKey(1)
        # 按下 's' 键保存
        if k == ord('s'):
            cnt += 1
            print(path_screenshots + "screenshot" + "_" + str(cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg")
            cv2.imwrite(path_screenshots + "screenshot" + "_" + str(cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg", img)

        # 按下 'q' 键退出
        if k == ord('q'):
            break

        cv2.imshow("camera", img)

# 释放摄像头
cap.release()
cv2.destroyAllWindows()
