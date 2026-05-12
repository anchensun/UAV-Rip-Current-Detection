import numpy as np
import cv2
import concurrent.futures
import os
import time
from OFlib import lucas_kanade, compute_flow_map, higher_order, addfile
from Typhoonmode_thread import Typhoon
from HornSchunck import HS

def run_vedio(core, frame_num, vediolib, mode):
    videos = os.listdir(vediolib)
    videos = filter(lambda x: x.endswith('mp4'), videos)
    for each_video in videos:
        print(each_video)
        each_video_name, _ = each_video.split('.')
        with concurrent.futures.ProcessPoolExecutor(core) as executor:
            cap = cv2.VideoCapture(vediolib + each_video)#'k:/Rip Videos/Sunset_hawaii_1.MOV')
            length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            #out = cv2.VideoWriter(video_save_path +'OpticalFlow_output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            firstFrame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
            count = 0
            cost = 0
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
                            # Bug fix (2026-05-03): np.int8 wraps |u| > 127 px/frame.
                            # See project_2020_code_audit.md.
                            u1_save = u1.astype(np.float32)
                            u2_save = u2.astype(np.float32)
                            np.savez(vediolib + 'result_of_frame' + str(count), u1_save, u2_save)
                            addfile(vediolib + '/' + each_video_name + '_typhoon.zip',
                                    vediolib + 'result_of_frame' + str(count) + '.npz')
                            os.remove(vediolib + 'result_of_frame' + str(count) + '.npz')

                    #SIMULTANEOUS HIGHER-ORDER OPTICAL FLOW ESTIMATION AND DECOMPOSITION
                    if mode == 'higher_order':
                        for u, v in executor.map(higher_order, b, p):
                            if u.shape == (10, 10):
                                break
                            print('No. ', count, ' frame has processed.')
                            count = count + 1
                            # Bug fix (2026-05-03): np.int8 wraps |u| > 127 px/frame.
                            u_save = u.astype(np.float32)
                            v_save = v.astype(np.float32)
                            np.savez(vediolib + 'result_of_frame' + str(count), u_save, v_save)
                            addfile(vediolib + '/' + each_video_name + '_LK_highorder.zip',
                                    vediolib + 'result_of_frame' + str(count) + '.npz')
                            os.remove(vediolib + 'result_of_frame' + str(count) + '.npz')
                            #flow_map = compute_flow_map(u, v)
                            #cv2.imshow('flow map', flow_map.astype(firstFrame.dtype))
                            #cv2.imwrite(video_save_path+str(count)+'.png',flow_map.astype(firstFrame.dtype));
                            #out.write(flow_map.astype(firstFrame.dtype))         
                 
                    # Baseline Optical Flow Mode
                    if mode == 'lucas_kanade':
                        for u, v in executor.map(lucas_kanade, b, p):
                            if u.shape == (10, 10):
                                break
                            print('No. ', count, ' frame has processed.')
                            count = count + 1
                            # Bug fix (2026-05-03): np.int8 wraps |u| > 127 px/frame.
                            u_save = u.astype(np.float32)
                            v_save = v.astype(np.float32)
                            np.savez(vediolib + 'result_of_frame' + str(count), u_save, v_save)
                            addfile(vediolib + '/' + each_video_name + '_LK_firstorder.zip',
                                    vediolib + 'result_of_frame' + str(count) + '.npz')
                            os.remove(vediolib + 'result_of_frame' + str(count) + '.npz')

                            #flow_map = compute_flow_map(u, v)
                            #cv2.imshow('flow map', flow_map.astype(firstFrame.dtype))
                            #cv2.imwrite(video_save_path+str(count)+'.png',flow_map.astype(firstFrame.dtype));

                            #update the previous and current frames
                            #previous_frame = current_frame
                            #current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            #out.write(flow_map.astype(firstFrame.dtype))

                    if mode == 'HS':
                        for u, v in executor.map(HS, b, p):
                            try:
                                if u.shape == (10, 10):
                                    break
                            except:
                                break
                            print('No. ', count, ' frame has processed.')
                            count = count + 1
                            # Bug fix (2026-05-03): np.int8 wraps |u| > 127 px/frame.
                            u_save = u.astype(np.float32)
                            v_save = v.astype(np.float32)
                            np.savez(vediolib + 'result_of_frame' + str(count), u_save, v_save)
                            addfile(vediolib + '/' + each_video_name + '_HS_firstorder.zip',
                                    vediolib + 'result_of_frame' + str(count) + '.npz')
                            os.remove(vediolib + 'result_of_frame' + str(count) + '.npz')

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
                    seconds = (cost * 0.7 + (stop - start) * 0.3) * (length - count) / frame_num
                    cost = stop - start
                    m, s = divmod(seconds, 60)
                    h, m = divmod(m, 60)
                    print(each_video_name, ' Mode: ', mode, ' Process:', '%.2f%%' % (100 * count / length), ' ETA: ','%d:%02d:%02d' % (h, m, s))
                else:
                    break

        cap.release()
        #cv2.destroyAllWindows()
        #out.release()

if __name__ == "__main__":
    core = 32     #Number of cores for processing.
    frame_num = 64 #Number of frames picked once.
    #vediolib = './OptOffTest/'#'./New_Rip_Video/' #Video Store Address
    #vediolib = './P_Optical/'
    #vediolib = './PalmBeachVideo/'
    #vediolib = './DJI_Video/'
    vediolib = './New_YouTube_Video/'
    #mode = 'typhoon' #Option for mode type
    #mode = 'lucas_kanade'
    mode = 'higher_order' #Change LK => HS; OFLib LK => HS; HighOrder 12normal => hornshunck
    #mode = 'HS'
    run_vedio(core,frame_num,vediolib,mode)
