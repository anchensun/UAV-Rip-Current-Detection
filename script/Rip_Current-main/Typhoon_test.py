import numpy as np
import cv2
import os
import time
from Typhoonmode import Typhoon
from OFlib import compute_flow_map

def run_test():
    videos = os.listdir(vediolib)
    videos = filter(lambda x: x.endswith('MOV'), videos)
    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        cap = cv2.VideoCapture(vediolib + each_video)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        background = cap.read()[1]
        firstFrame = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        secondFrame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
        count = 0
        for i in range(36):
            for j in range(i+1):
                count = count + 1
                testget(1000., None, (i+1)*4, (j+1)*4, count, background, firstFrame, secondFrame, frame_width, frame_height)
        break

def testget(a, b, c, d, testset, background, firstFrame, secondFrame, frame_width, frame_height):
    start = time.time()
    (u, v), _ = Typhoon.solve_pyramid(firstFrame, secondFrame, a, b, c, d)
    stop = time.time()
    seconds = (stop - start)
    print('The test cost seconds:', seconds)
    u.astype(float)
    uint8 = np.int8(u)
    v.astype(float)
    vint8 = np.int8(v)
    flow_map = compute_flow_map(uint8, vint8)
    mask = cv2.resize(flow_map.astype(background.dtype), (frame_width, frame_height))
    maskrgb = np.array([mask, mask, mask])
    maskrgb = maskrgb.swapaxes(0, 1)
    maskrgb = maskrgb.swapaxes(1, 2)
    dst = 255 - maskrgb
    img = dst & background
    cv2.imwrite(vediolib + 'Factr 1000 Maxfun ' + str(c) + ' Maxiter' + str(d) + ' Seconds:' + str(int(seconds)) + '.jpg', img)

if __name__ == "__main__":
    vediolib = '/home/anchen/Near_Shore_Wave_Speed_Estimation/Rip_Video/' #Video Store Address
    run_test()
