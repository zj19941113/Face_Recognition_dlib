# Face_Recognition_dlib
## 环境  
1. Ubuntu 16.04  
2. opencv 3.0 for python3.6 `pip install opencv-python`  
3. dlib 19.16  

编译dlib库出现问题，可参考这篇：https://blog.csdn.net/ffcjjhv/article/details/84660869  

## 模型下载  
人脸关键点检测器 `predictor_path="shape_predictor_68_face_landmarks.dat`  
人脸识别模型 `face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat`  
含人脸库candidate-face中人脸不同表情的测试数据集 `test-face.zip` 解压后与上述文件均置于根目录下  
下载地址 ： 百度云盘 https://pan.baidu.com/s/1h01sfvf5KWU6_7c2-i5HTQ  
## 运行  
运行 `python candidate_train.py` 获得人脸库特征信息，存储在`candidates.npy` 与 `candidates.txt` 中  
运行 `python facerec_68point.py`  得到识别结果all-face-result.jpg  
运行 `this_is_who.py` 进行在`test-face`文件夹中的批量测试，测试结果存于`faceRec`文件夹，识别错误结果存于`faceRec_ERROR` ,不设相似度阈值时识别正确率为0.99469，但是这里寻找的是与数据库中最相似的人脸，加入相似度阈值使非数据库中的人显示unknown，相似度阈值is_not_candidate=0.5时，准确率0.976127，相似度阈值is_not_candidate=0.6时，准确率0.986737，但是将unknow识别为人脸库人脸的可能性会升高。  
运行 `this_is_who_camera.py`  打开摄像头进行实时的人脸识别  
## 补充    
1. 每次人脸库candidate-face中加入新的人脸数据，均需运行`python candidate_train.py` 
2. 最近的项目是在红外人脸图像上进行的，人脸不太清晰，如果是正常摄像头效果应该会更好
## 运行结果  
![](https://github.com/zj19941113/Face_Recognition_dlib/blob/master/img/faces.JPG) 
![](https://github.com/zj19941113/Face_Recognition_dlib/blob/master/img/result2.png) 
![](https://github.com/zj19941113/Face_Recognition_dlib/blob/master/Animation.gif)   
![](https://github.com/zj19941113/Face_Recognition_dlib/blob/master/screenShots/screenshot_4_2018-12-13-17-19-42.jpg) 
