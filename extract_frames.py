import numpy as np
import cv2
import matplotlib.pyplot as plt
path = 'imgs/'
# read video
cap = cv2.VideoCapture('0X1A5FAE3F9D37794E.avi')
frames=[]
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    frames.append(cv2.resize(frame,(900,900),interpolation = cv2.INTER_CUBIC))
cap.release()
for i in range(2):
    cv2.imwrite(path + 'frame{}.png'.format(i), frames[i])

