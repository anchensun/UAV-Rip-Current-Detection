import numpy as np
import cv2 as cv
import os

vediolib = './DJI_Video/'
videos = os.listdir(vediolib)
videos = filter(lambda x: x.endswith('mp4'), videos)
for each_video in videos:
    print(each_video)
    cap = cv.VideoCapture(cv.samples.findFile(vediolib + each_video))
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    i = 0
    while(i < 60):
        i += 1 
        ret, frame2 = cap.read()
        if not ret:
            print('No frames grabbed!')
            break
        next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 1] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        #k = cv.waitKey(30) & 0xff
        #if k == 27:
        #    break
        #elif k == ord('s'):
        #   #cv.imwrite('opticalfb.png', frame2)
        cv.imwrite(vediolib + each_video.split('.')[0] + '_opticalhsv.png', bgr)
        prvs = next
