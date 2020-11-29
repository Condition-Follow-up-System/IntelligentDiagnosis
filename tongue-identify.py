import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("c:/Users/16521/Desktop/shape_predictor_68_face_landmarks.dat")


# cv2读取图像
img = cv2.imread("c:/test1.jpg")

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
xxList = []
yyList = []
maxX = 0
minX = 0
maxY = 0
minY = 0
# 人脸数rects
rects = detector(img_gray, 0)
for i in range(len(rects)):
     # 特征提取
     landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])
     for idx, point in enumerate(landmarks):
        if idx > 49:
            pos = (point[0, 0], point[0, 1])
            x = point[0, 0],
            y = point[0, 1]
            xList = []
            yList = []
            xList.append(x)  # 添加数据
            yList.append(y)
            for k in xList:
                list(k)
                xxList.append(k[0])
            # for l in xxList:
                # print(l)
            maxX = max(xxList)
            minX = min(xxList)
            # print(maxX,minX)


            for p in yList:
                # list(p)
                yyList.append(p)
                # yyList.append(j[0])
                #print(p)
            maxY = max(yyList)
            minY = min(yyList)






            # print(idx + 1, pos)
            # 利用cv2.circle给每个特征点画一个圈，共68个
            #cv2.circle(img, pos, 2, color=(0, 255, 0))
            # 利用cv2.putText输出1-6

            font = cv2.FONT_HERSHEY_SIMPLEX

            # cv2.putText(img, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)pts=np.array([[10,3],[48,
            # 19],[60,3],[98,19]],np.int32) #数据类型必须是int32
# pts=pts.reshape((-1,1,2))
cv2.rectangle(img, (minX-10, maxY+20), (maxX+10, minY-10), (255, 255, 255), 1)  # 闭合矩形
#print(maxY, minY)
# cv2.polylines(img,[pts],True,(0,0,255),1) # 图像，点集，是否闭合，颜色，线条粗细
cv2.namedWindow("img", 2)
cv2.imshow("img", img)
cv2.waitKey(0)

