import numpy as np
import cv2
import dlib
detector=dlib.get_frontal_face_detector()
img=cv2.imread('c:/tyy.jfif')
gray=cv2.cvtColor(src=img,code=cv2.COLOR_BGR2GRAY)
faces=detector(gray)
for face in faces:
	x1=face.left()
	y1=face.top()
	x2=face.right()
	y2=face.bottom()
	cv2.rectangle(img=img,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,0),thickness=4)
cv2.imshow(winname="Face",mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()






