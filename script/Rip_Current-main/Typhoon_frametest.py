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
        for k in range(60):
            secondback = cap.read()[1]
            secondFrame = cv2.cvtColor(secondback, cv2.COLOR_BGR2GRAY)
            testget(1000., None, 60, 60, k+1, background, firstFrame, secondFrame, frame_width, frame_height)
            testget(1000., None, 60, 40, k+1, background, firstFrame, secondFrame, frame_width, frame_height)
            testget(1000., None, 60, 20, k+1, background, firstFrame, secondFrame, frame_width, frame_height)
            testget(1000., None, 40, 20, k+1, background, firstFrame, secondFrame, frame_width, frame_height)
            testget(1000., None, 40, 40, k+1, background, firstFrame, secondFrame, frame_width, frame_height)
            testget(1000., None, 80, 80, k+1, background, firstFrame, secondFrame, frame_width, frame_height)
            testget(1000., None, 80, 60, k+1, background, firstFrame, secondFrame, frame_width, frame_height)
            testget(1000., None, 80, 40, k+1, background, firstFrame, secondFrame, frame_width, frame_height)
            testget(1000., None, 80, 20, k+1, background, firstFrame, secondFrame, frame_width, frame_height)
            firstFrame = secondFrame
            background = secondback
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
    cv2.imwrite(vediolib + '/Frametest/Frame: ' + str(testset) + ' Maxfun ' + str(c) + ' Maxiter' + str(d) + ' Seconds:' + str(int(seconds)) + '.jpg', img)

if __name__ == "__main__":
    vediolib = '/home/anchen/Near_Shore_Wave_Speed_Estimation/Rip_Video/' #Video Store Address
    run_test()
