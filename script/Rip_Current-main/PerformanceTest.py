    
import numpy as np
import cv2
import matplotlib
import concurrent.futures
import os
import time
from OFlib import lucas_kanade, compute_flow_map, higher_order, addfile
from Typhoonmode_thread import Typhoon

def run_vedio(core, frame_num, vediolib, mode,testround):
    videos = os.listdir(vediolib)
    videos = filter(lambda x: x.endswith('MOV'), videos)
    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        #os.mkdir(vediolib + '/' + each_video_name + 'optical flow output')
        #video_save_path = vediolib + '/' + each_video_name + 'optical flow output' + '/'
        #with gzip.open(vediolib+'/'+each_video_name+'_result.gz', 'wb') as f:
            #f.write('')
        with concurrent.futures.ProcessPoolExecutor(core) as executor:
            #frame_width = 1913
            #frame_height = 1073
            #fourcc = cv2.cv.CV_FOURCC(*'XVID')
            cap = cv2.VideoCapture(vediolib + each_video)#'k:/Rip Videos/Sunset_hawaii_1.MOV')
            length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            #out = cv2.VideoWriter(video_save_path +'OpticalFlow_output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            firstFrame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
            #cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            #cv2.namedWindow("flow map", cv2.WINDOW_NORMAL)
            #current_frame = firstFrame
            #previous_frame = current_frame
            count = 0
            cost = 0
            roundcount = 0
            while True:
                roundcount = roundcount + 1
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

                    #atr = 0
                    #for item in b:
                    #    if not(hasattr(item, 'shape')):
                    #        del b[atr]
                    #    atr = atr + 1
                    #atr = 0
                    #for item in p:
                    #    if not (hasattr(item, 'shape')):
                    #        del p[atr]
                    #    atr = atr + 1
                    #Start CPU parallel computing:

                    #Typhoon Mode
                    if mode == 'typhoon':
                        for x in executor.map(Typhoon.solve_pyramid, zip(b, p)):
                            u1 = x[0]
                            u2 = x[1]
                            print('No. ', count, ' frame has processed.')
                            count = count + 1
                            u1.astype(float)
                            u1int8 = np.int8(u1)
                            u2.astype(float)
                            u2int8 = np.int8(u2)
                            
                    #SIMULTANEOUS HIGHER-ORDER OPTICAL FLOW ESTIMATION AND DECOMPOSITION
                    if mode == 'higher_order':
                        for u, v in executor.map(higher_order, b, p):
                            if u.shape == (10, 10):
                                break
                            print('No. ', count, ' frame has processed.', end = '\r')
                            count = count + 1
                            u.astype(float)
                            uint8 = np.int8(u)
                            v.astype(float)
                            vint8 = np.int8(v)
                            
                            #flow_map = compute_flow_map(u, v)
                            #cv2.imshow('flow map', flow_map.astype(firstFrame.dtype))
                            #cv2.imwrite(video_save_path+str(count)+'.png',flow_map.astype(firstFrame.dtype));
                            #out.write(flow_map.astype(firstFrame.dtype))         
                 
                    # Baseline Optical Flow Mode
                    if mode == 'lucas_kanade':
                        for u, v in executor.map(lucas_kanade, b, p):
                            if u.shape == (10, 10):
                                break
                            print('No. ', count, ' frame has processed.', end = '\r')
                            count = count + 1
                            # cv2.imshow('frame',frame)

                            # finding the optical flow between two consecutive frames
                            u.astype(float)
                            uint8 = np.int8(u)
                            v.astype(float)
                            vint8 = np.int8(v)

                            #flow_map = compute_flow_map(u, v)
                            #cv2.imshow('flow map', flow_map.astype(firstFrame.dtype))
                            #cv2.imwrite(video_save_path+str(count)+'.png',flow_map.astype(firstFrame.dtype));

                            #update the previous and current frames
                            #previous_frame = current_frame
                            #current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            #out.write(flow_map.astype(firstFrame.dtype))

                    #if cv2.waitKey(30) & 0xFF == ord('q'):
                    #    break
                    stop = time.time()
                    #seconds = (cost * 0.7 + (stop - start) * 0.3) * (length - count) / frame_num
                    cost = stop - start
                    #m, s = divmod(seconds, 60)
                    #h, m = divmod(m, 60)
                    f = open( vediolib + 'PerformanceTest.txt', 'a')
                    f.write('\n' + each_video_name + ' Mode: ' + mode + ' Cost Time ' + str(cost))
                    f.close()
                    print(each_video_name, ' Mode: ', mode, ' Cost Time: ', cost, end = '\r')
                    
                    if roundcount == testround:
                        print(each_video_name, ' Mode: ', mode, ' has tested.')
                        break

                else:
                    break

        cap.release()
        #cv2.destroyAllWindows()
        #out.release()

if __name__ == "__main__":
    core = 1      #Number of cores for processing.
    frame_num = 8 #Number of frames picked for once.
    testround = 5  #Round of test
    vediolib = '/home/anchen/Near_Shore_Wave_Speed_Estimation/PerformanceTestVideo/' #Video Store Address
    #mode = 'typhoon' #Option for mode type
    mode = 'lucas_kanade'
    #mode = 'higher_order'
    f = open(vediolib + 'PerformanceTest.txt', 'w')
    f.write('Performance Test in Environment: Core = ' + str(core) + ' Frame_num = ' + str(frame_num) + ' Test Round = ' + str(testround))
    f.close()
    run_vedio(core,frame_num,vediolib,mode,testround)
    mode = 'higher_order'
    #frame_num = 4
    core = 2
    f = open(vediolib + 'PerformanceTest.txt', 'a')
    f.write('Performance Test in Environment: Core = ' + str(core) + ' Frame_num = ' + str(frame_num) + ' Test Round = ' + str(testround))
    f.close()
    run_vedio(core,frame_num,vediolib,mode,testround)
