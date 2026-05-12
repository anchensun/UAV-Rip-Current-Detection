import numpy as np
import cv2
import matplotlib
#from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ThreadPool
import os
import time
import profile
from OFlib import lucas_kanade, compute_flow_map, higher_order, addfile
from Typhoonmode_thread import Typhoon

def run_vedio(core, frame_num, vediolib, mode):
    videos = os.listdir(vediolib)
    videos = filter(lambda x: x.endswith('MOV'), videos)
    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        if each_video:
            pool = ThreadPool(core)
            cap = cv2.VideoCapture(vediolib + each_video)
            length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            firstFrame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
            count = 0
            while True:
                start = time.time()
                ret = [[]] * frame_num
                frame = [[]] * frame_num
                b = [[]] * (frame_num)
                p = [[]] * (frame_num)
                frameprocess = 0
                for i in range(frame_num):
                    ret[i], frame[i] = cap.read()
                    if ret[i] == False:
                        frameprocess = i - 1
                        break
                b[0] = firstFrame
                if frameprocess == 0:
                    frameprocess = frame_num - 1
                if ret[0] == True:
                    for i in range(frameprocess):
                        b[i + 1] = cv2.cvtColor(frame[i], cv2.COLOR_BGR2GRAY)
                        p[i] = cv2.cvtColor(frame[i], cv2.COLOR_BGR2GRAY)
                    p[frameprocess] = cv2.cvtColor(frame[frameprocess], cv2.COLOR_BGR2GRAY)
                    firstFrame = p[frameprocess]
                    loading = time.time()
                    seconds = (loading - start)
                    print('Loading Cost: ', seconds)     
            
                    #Typhoon Mode
                    if mode == 'typhoon':
                        for (u1, u2) in pool.imap(Typhoon.solve_pyramid, zip(b, p)):
                            if u1.shape == (10, 10):
                                break
                            print('No. ', count, ' frame has processed.')
                            count = count + 1
                            # Bug fix (2026-05-03): np.int8 wraps |u| > 127 px/frame.
                            u1_save = u1.astype(np.float32)
                            u2_save = u2.astype(np.float32)
                            np.savez(vediolib + 'result_of_frame' + str(count), u1_save, u2_save)
                            addfile(vediolib + '/' + each_video_name + '_typhoon_result_typhoon_mode.zip',
                                    vediolib + 'result_of_frame' + str(count) + '.npz')
                            os.remove(vediolib + 'result_of_frame' + str(count) + '.npz')

                    stop = time.time()
                    seconds = (stop - start) * (length - count) / frame_num
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    print('Process:', '%.2f%%' % (100 * count / length), ' ETA: ','%d:%02d:%02d' % (h, m, s))              
                else:
                    break

        cap.release()
        
if __name__ == "__main__":
    core = 16      #Number of cores for processing.
    frame_num = 16 #Number of frames picked once.
    vediolib = '/home/anchen/Near_Shore_Wave_Speed_Estimation/Rip_Video/' #Video Store Address
    mode = 'typhoon' #Option for mode type
    run_vedio(core,frame_num,vediolib,mode)
