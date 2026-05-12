import numpy as np
import cv2
import matplotlib.pyplot as plt
import zipfile
import os
import concurrent.futures
import time
import random
from OFlib import compute_flow_map, digitaldye, heatmap, un_zip

def gen_vedio(sec, core, vediolib, mode, gaptime, merge):
    videos = os.listdir(vediolib)
    videos = filter(lambda x: x.endswith('zip'), videos)
    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        with concurrent.futures.ProcessPoolExecutor(core) as executor:
            # cv2.namedWindow("flow map", cv2.WINDOW_NORMAL)
            #if os.path.isdir(vediolib + each_video_name):
            #    pass
            #else:
            #    os.mkdir(vediolib + each_video_name)
            videoname = findvideoname(vediolib, each_video_name)
            cap = cv2.VideoCapture(vediolib + videoname + '.MOV')
            rate = int(cap.get(cv2.CAP_PROP_FPS))
            #print(rate)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
            # print(rate,frame_height,frame_width)
            out = cv2.VideoWriter(vediolib + mode + '_' + each_video_name + '.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                                  rate, (frame_width * merge, frame_height), 1)
            # firstFrame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
            framezip = zipfile.ZipFile(vediolib + each_video)
            frame_names = framezip.namelist()
            i = 0
            cost = 0
            length = len(frame_names)
            if length > (sec * rate):
                length = sec * rate
 
            for name in frame_names:
                start = time.time()
                i = i + 1
                background = cap.read()[1]

                if ((i / rate) % 5 == 0):
                    cv2.imwrite('/a/bear.cs.fiu.edu./disk/bear-c/users/asun/Unet/TGRS_Test/RipCurrent/' + videoname + '_sec' + str(int(i / rate)) + '.jpg', background)
                if i > sec * rate:
                    cap.release()
                    out.release()
                    break
                stop = time.time()
                seconds = (cost * 0.7 + (stop - start) * 0.3) * (length - i)
                cost = stop - start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                print(each_video_name, ' frame ', str(i), ' Process:', '%.2f%%' % (100 * i / length), ' ETA: ',
                      '%d:%02d:%02d' % (h, m, s), end = '\r')
            cap.release()
            out.release()

def findvideoname(video_dir, framename):
    videos = os.listdir(video_dir)
    videos = filter(lambda x: x.endswith('MOV'), videos)
    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        if each_video_name == framename[0:len(each_video_name)]:
            return each_video_name

def motionmask(frame, u, v):
    newu = np.zeros(u.shape)
    newv = np.zeros(v.shape)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 50])
    upper_blue = np.array([50, 50, 225])
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([225, 25, 255])
    mask2 = cv2.inRange(hsv, lower_white, upper_white)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 100])
    mask3 = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.add(mask1, mask3)
    
    grandmaskframe = cv2.imread(vediolib + 'grandmask.jpg')
    grandmask = cv2.cvtColor(grandmaskframe, cv2.COLOR_BGR2GRAY)
    mask = cv2.add(mask, grandmask)
    
    mask = cv2.add(mask, mask2)
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    # cv2.imwrite(vediolib + 'try1.jpg', mask1)
    # cv2.imwrite(vediolib + 'try2.jpg', mask2)
    # cv2.imshow('res', res)
    for y in range(u.shape[1]):
        for x in range(u.shape[0]):
            if mask[x, y] == 0:
                newu[x, y] = u[x, y]
                newv[x, y] = v[x, y]
    return newu, newv

def heatmask(frame, heatmap):
    newheat = np.zeros(heatmap.shape)
    newframe = cv2.inRange(frame , [0, 0, 0], [200, 200, 200])
    hsv = cv2.cvtColor(newframe, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 50])
    upper_blue = np.array([50, 50, 225])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # res = cv2.bitwise_and(frame, frame, mask=mask)
    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    # cv2.imwrite(vediolib + 'try.jpg', mask)
    # cv2.imshow('res', res)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] == 0:
                newheat[x, y] = heatmap[x, y]
    return newheat

def plotframe(u ,v, each_video_name, name, i):
    fig, ax = plt.subplots()
    q = ax.quiver(u, v)
    ax.quiverkey(q, X=0.1, Y=0.1, U=1, label='Test Demo', labelpos='E')
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(vediolib + each_video_name + '/' + '/frame' + str(i) + '.png')
    return
    
if __name__ == "__main__":
    sec = 1000      #Seconds of Demo Video
    core = 8      #Number of cores for processing.
    gaptime = 0   #Seconds of Gap Time
    merge = 2     #"2" is merge mode, "1" is normal mode.
    vediolib = './Rip_Save/' #Video Store Address
    #mode = 'demo'
    #mode = 'dye'
    mode = 'heatmap'
    gen_vedio((sec + gaptime), core, vediolib, mode, gaptime, merge)
