import numpy as np
import cv2
import matplotlib.pyplot as plt
import zipfile
import os
import concurrent.futures
import time
import random
from math import sqrt
import math
from OFlib import compute_flow_map, digitaldye, heatmap, un_zip
from OptOffShoreDirection import OffshoreDir

def gen_vedio(sec, core, frame_num, vediolib, mode, gaptime, merge, vide_type, rate, superresolution = 1):
    videos = os.listdir(vediolib)
    videos = filter(lambda x: x.endswith('zip'), videos)
    for each_video in videos:
        #heat_total = np.zeros([frame_width, frame_height])
        mask_flag = 0
        print('Process: ', each_video, end = '\n')
        each_video_name, _ = each_video.split('.')
        with concurrent.futures.ProcessPoolExecutor(core) as executor:
            if os.path.isdir(vediolib + each_video_name):
                pass
            else:
                os.mkdir(vediolib + each_video_name)
            videoname = findvideoname(vediolib, each_video_name, vide_type)
            cap = cv2.VideoCapture(vediolib + videoname + '.' + vide_type)
            #rate = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));
            #out = cv2.VideoWriter(vediolib + mode + '_' + each_video_name + '.mp4', cv2.VideoWriter_fourcc(*'MP4V'), rate, (frame_width * merge, frame_height), 1)
            framezip = zipfile.ZipFile(vediolib + each_video)
            frame_names = framezip.namelist()
            i = 0
            cost = 0
            length = len(frame_names)
            print('Length: ', length, end = '\n')

            heat_total = np.zeros([frame_width, frame_height])
            mask_total = np.zeros([frame_width, frame_height])

            if length > (sec * rate):
                length = sec * rate

            if mode == 'heatmap':
                heat_list = []
                model_0_list = []
                model_1_list = []
                model_2_list = []

                off_u_list = []
                off_v_list = []

                videoname_list = []

                pers_center_list = []

                offdirname = vediolib + videoname + '_OffShoreDir_1by1.npz'
                mixedoffdirname = vediolib + videoname + '_OffShoreDir_Mixed.npz'
                if os.path.exists(mixedoffdirname):
                    mixedoffdirectionget = np.load(mixedoffdirname)
                    off_v, off_u = mixedoffdirectionget["arr_0"], mixedoffdirectionget["arr_1"]
                elif os.path.exists(offdirname) and superresolution == 1:
                    offdirectionget = np.load(offdirname)
                    off_v, off_u = offdirectionget["arr_0"], offdirectionget["arr_1"]
                else:  
                    offdirectionget = np.load(offdirname)
                    off_v_tmp, off_u_tmp = offdirectionget["arr_0"], offdirectionget["arr_1"]
                    off_u = np.zeros((int(off_u_tmp.shape[0] / superresolution),
                                      int(off_u_tmp.shape[1] / superresolution)))
                    off_v = np.zeros((int(off_v_tmp.shape[0] / superresolution),
                                      int(off_v_tmp.shape[1] / superresolution)))
                    for i in range(int(off_u_tmp.shape[0] / superresolution)):
                        for j in range(int(off_u_tmp.shape[1] / superresolution)):   
                            u_sum = 0
                            v_sum = 0
                            for k in range(superresolution):    
                                for q in range(superresolution):    
                                    u_sum += off_u_tmp[i * superresolution + k, j * superresolution + q]
                                    v_sum += off_v_tmp[i * superresolution + k, j * superresolution + q]
                            off_u[i, j] = int(u_sum / (superresolution * superresolution))
                            off_v[i, j] = int(v_sum / (superresolution * superresolution))
           
                heat = np.zeros([frame_width, frame_height])

                #Check whether skyline is available
                grandmaskframe = cv2.cvtColor(cv2.imread('./New_Rip_Mask/' + videoname + '_mask.png'), cv2.COLOR_BGR2GRAY)
                CoLine_list, SkyLine_list = find_skyline_mid(grandmaskframe)
                if os.path.exists(mixedoffdirname) :
                    off_u_mix = off_u
                    off_v_mix = off_v
                    mid_point = [frame_height - 1, int(frame_width / 2 - 1)]
                    perspective_point = find_pres_point(SkyLine_list, mid_point)
                elif SkyLine_list == []:
                    off_u_mix = off_u
                    off_v_mix = off_v
                    perspective_point = [0, 0]
                else:
                    mid_point = [frame_height - 1, int(frame_width / 2 - 1)]
                    perspective_point = find_pres_point(SkyLine_list, mid_point)
                    off_u_mix = np.zeros((off_u.shape[0], off_u.shape[1]))  
                    off_v_mix = np.zeros((off_v.shape[0], off_v.shape[1]))
                    
                    #vision_dis = 112000
                    for i in range(SkyLine_list[0][0], off_u.shape[0]):
                        begin_point = [i, 0]
                        if i == perspective_point[0]:
                            cal_points_list = []
                            for j in range(perspective_point[1]):
                                cal_points_list.append([i, j])
                        else:
                            cal_points_list = find_points(np.array([begin_point, perspective_point])).tolist()
                        dy = perspective_point[0] - i
                        dx = perspective_point[1]
                        if i <= perspective_point[0]:
                            for points in cal_points_list:
                                off_u_mix[points[0], points[1]] = dx
                                off_v_mix[points[0], points[1]] = dy
                        else:
                            begin_index = find_max_dis(cal_points_list, CoLine_list)
                            if begin_index == 0:
                                unit_distance = sqrt((off_u.shape[0] - perspective_point[0] - 1) ** 2 + (dx * (off_u.shape[0] - perspective_point[0] - 1) / (i - perspective_point[0])) ** 2)
                            else:
                                unit_distance = sqrt((cal_points_list[begin_index][0] - perspective_point[0]) ** 2 + (cal_points_list[begin_index][1] - perspective_point[1]) ** 2)
                            for j in range(begin_index, len(cal_points_list)):
                                distance = sqrt((cal_points_list[j][0] - perspective_point[0]) ** 2 + (cal_points_list[j][1] - perspective_point[1]) ** 2)
                                ratio = (distance / unit_distance) ** 2
                                #print(ratio)
                                points = cal_points_list[j]
                                dx_off = off_u[points[0], points[1]]
                                dy_off = off_v[points[0], points[1]]

                                off_value = sqrt(dx_off ** 2 + dy_off ** 2)
                                pes_value = sqrt(dx ** 2 + dy ** 2)
                                if off_value != 0:
                                    dx_resize = dx / (pes_value / off_value)
                                    dy_resize = dy / (pes_value / off_value)
                                else:
                                    dx_resize = dx
                                    dy_resize = dy
                                #print(dx_resize, dy_resize)
                                off_u_mix[points[0], points[1]] = dx_resize * (1 - ratio) + ratio * dx_off
                                off_v_mix[points[0], points[1]] = dy_resize * (1 - ratio) + ratio * dy_off
                                #print(cal_points_list[j], dx_resize, dx_off, dy_resize, dy_off, azimuthAngle(points[0], points[1], dx, dy), azimuthAngle(points[0], points[1], off_u_mix[points[0], points[1]], off_v_mix[points[0], points[1]]))

                    for i in range(SkyLine_list[-1][0], off_u.shape[0]):
                        begin_point = [i, off_u.shape[1] - 1]
                        if i == perspective_point[0]:
                            cal_points_list = []
                            for j in range(off_u.shape[0] - 1, perspective_point[1], -1):
                                cal_points_list.append([i, j])
                        else:
                            cal_points_list = find_points(np.array([begin_point, perspective_point])).tolist()
                        dy = perspective_point[0] - i
                        dx = perspective_point[1] + 1 - off_u.shape[1]
                        #print(dx, dy)
                        if i <= perspective_point[0]:
                            for points in cal_points_list:
                                off_u_mix[points[0], points[1]] = dx
                                off_v_mix[points[0], points[1]] = dy
                        else:
                            begin_index = find_max_dis(cal_points_list, CoLine_list)
                            if begin_index == 0:
                                unit_distance = sqrt((off_u.shape[0] - perspective_point[0] - 1) ** 2 + (dy * (off_u.shape[0] - perspective_point[0] - 1) / (i - perspective_point[0])) ** 2)
                            else:
                                unit_distance = sqrt((cal_points_list[begin_index][0] - perspective_point[0]) ** 2 + (cal_points_list[begin_index][1] - perspective_point[1]) ** 2)
                            for j in range(begin_index, len(cal_points_list)):
                                distance = sqrt((cal_points_list[j][0] - perspective_point[0]) ** 2 + (cal_points_list[j][1] - perspective_point[1]) ** 2)
                                ratio = (distance / unit_distance) ** 2
                                points = cal_points_list[j]
                                dx_off = off_u[points[0], points[1]]
                                dy_off = off_v[points[0], points[1]]
                                off_value = sqrt(dx_off ** 2 + dy_off ** 2)
                                if off_value != 0:
                                    dx_resize = dx / (pes_value / off_value)
                                    dy_resize = dy / (pes_value / off_value)
                                else:
                                    dx_resize = dx
                                    dy_resize = dy
                                off_u_mix[points[0], points[1]] = dx_resize * (1 - ratio) + ratio * dx_off
                                off_v_mix[points[0], points[1]] = dy_resize * (1 - ratio) + ratio * dy_off
                                #print(cal_points_list[j], dx_resize, dx_off, dy_resize, dy_off, azimuthAngle(points[0], points[1], dx_resize, dy_resize), azimuthAngle(points[0], points[1], dx_off, dy_off), azimuthAngle(points[0], points[1], off_u_mix[points[0], points[1]], off_v_mix[points[0], points[1]]))

                    for i in range(off_u.shape[1]):
                        begin_point = [off_u.shape[0] - 1, i]
                        if i == perspective_point[1]:
                            cal_points_list = []
                            for j in range(off_u.shape[0] - 1, perspective_point[0], -1):
                                cal_points_list.append([j, i])
                        else:
                            cal_points_list = find_points(np.array([begin_point, perspective_point])).tolist()
                        #if i <= perspective_point[1]:
                        dy = perspective_point[0] + 1 - off_u.shape[0]
                        dx = perspective_point[1] - i
                        #else:
                        #    dx = perspective_point[0] - off_u.shape[0]
                        #    dy = perspective_point[1] - i
                        begin_index = find_max_dis(cal_points_list, CoLine_list)
                        if begin_index == 0:
                            unit_distance = sqrt(dx ** 2 + dy ** 2)
                        else:
                            unit_distance = sqrt((cal_points_list[begin_index][0] - perspective_point[0]) ** 2 + (cal_points_list[begin_index][1] - perspective_point[1]) ** 2)
                        for j in range(begin_index):
                            points = cal_points_list[j]
                            off_u_mix[points[0], points[1]] = off_u[points[0], points[1]]
                            off_v_mix[points[0], points[1]] = off_v[points[0], points[1]]
                        for j in range(begin_index, len(cal_points_list)):
                            distance = sqrt((cal_points_list[j][0] - perspective_point[0]) ** 2 + (cal_points_list[j][1] - perspective_point[1]) ** 2)
                            ratio = (distance / unit_distance) ** 2
                            points = cal_points_list[j]
                            dx_off = off_u[points[0], points[1]]
                            dy_off = off_v[points[0], points[1]]
                            off_value = sqrt(dx_off ** 2 + dy_off ** 2)
                            if off_value != 0:
                                dx_resize = dx / (pes_value / off_value)
                                dy_resize = dy / (pes_value / off_value)
                            else:
                                dx_resize = dx
                                dy_resize = dy
                            off_u_mix[points[0], points[1]] = dx_resize * (1 - ratio) + ratio * dx_off
                            off_v_mix[points[0], points[1]] = dy_resize * (1 - ratio) + ratio * dy_off

                    flow_map = compute_flow_map(off_u_mix, off_v_mix)
                    cv2.imwrite(vediolib + 'mixedoffdirflowmap.jpg', flow_map)
                    # Bug fix (2026-05-03): was np.int8 — distance-to-coast vectors
                    # easily exceed 127 px on a 1920x1080 frame.
                    u_save = off_u_mix.astype(np.float32)
                    v_save = off_v_mix.astype(np.float32)
                    np.savez(vediolib + videoname + '_OffShoreDir_Mixed', u_save, v_save)

                for i in range(frame_num):
                    heat_list.append(heat)
                    model_0_list.append(0)
                    model_1_list.append(1)
                    model_2_list.append(2)
                    off_u_list.append(off_u_mix) 
                    off_v_list.append(off_v_mix)
                    videoname_list.append(videoname)
                    pers_center_list.append(perspective_point)
 
            for i in range(0, length, frame_num):
                start = time.time()
                name_list = []
                u_list = []
                v_list = []
                background_list = []
                mask_list = []
                for j in range(frame_num):
                    name = frame_names[i + j]
                    fileget = framezip.open(name)
                    arrayget = np.load(fileget)
                    u = arrayget["arr_0"]
                    v = arrayget["arr_1"]
                    background = cap.read()[1]
                    u_list.append(u)
                    v_list.append(v)
                    background_list.append(background)

                get_u_list = []
                get_v_list = []
                for get_u, get_v, p_count in executor.map(motionmask, background_list, u_list, v_list, videoname_list):
                    get_u_list.append(get_u)
                    get_v_list.append(get_v)
                    mask_total += p_count   
                
                if mode == 'heatmap': # 1: skip; 0: plot color heatmap; 2: only calculate
                    if (i < gaptime * rate):
                        flow_map, empty = heatmap(u, v, off_u, off_v, frame_width, frame_height, heat, 1, perspective_point)
                    else:
                        for heat in executor.map(heatmap, get_u_list, get_v_list, off_u_list, off_v_list, heat_list, model_2_list, pers_center_list):
                            heat_total += heat 
                        #print(i, ' Heat')
                        #heat_total_int8 = np.int8(heat_total)
                        #np.savez(vediolib + 'FinalPlot_' + str(sec) + 's/' + 'RipDetection_' + each_video_name, heat_total_int8)
                stop = time.time()
                seconds = (cost * 0.7 + (stop - start) * 0.3) * (length - i) / frame_num
                cost = stop - start
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                print(each_video_name, ' frame ', str(i), ' Process:', '%.2f%%' % (100 * i / length), ' ETA: ', '%d:%02d:%02d' % (h, m, s), end = '\r')
            
                #Save per rate * 2 frame
                if i % (rate * 2) == 0 and i != 0:
                    flow_map, empty = heatmap(get_u, get_v, off_u, off_v, heat_total, 3, perspective_point, i)
                    mask = cv2.resize(flow_map.astype(background.dtype), (frame_width, frame_height))
                    #heat_total_int8 = np.int8(heat_total)
                    np.savez(vediolib + 'FinalPlot/' + 'RipDetection_' + str(i) + '_' + each_video_name, heat_total)
                    np.savez(vediolib + 'FinalPlot/' + 'RipMaskCount_' + str(i) + '_' + each_video_name, mask_total)
                    if merge == 2:
                        img = np.hstack((mask, background))
                    else:
                        maskgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                        ret, maskmask = cv2.threshold(maskgray, 10, 255, cv2.THRESH_BINARY)
                        mask_inv = cv2.bitwise_not(maskmask)
                        background_bg = cv2.bitwise_and(background, background, mask = mask_inv)
                        mask_fg = cv2.bitwise_and(mask, mask, mask = maskmask)
                        img = cv2.add(background_bg, mask_fg)

                    cv2.imwrite(vediolib + 'FinalPlot/' + each_video_name + '_' + str(i) + '_finalplot.jpg', img)                
    
            #Plot Heatmap
            #heat_total_int8 = np.int8(heat_total)
            np.savez(vediolib + 'FinalPlot_' + str(sec) + 's/' + 'RipDetection_' + each_video_name, heat_total)
            np.savez(vediolib + 'FinalPlot_' + str(sec) + 's/' + 'RipMaskCount_' + str(i) + '_' + each_video_name, mask_total)

            flow_map, empty = heatmap(get_u, get_v, off_u, off_v, heat_total, 3, perspective_point, length)
            mask = cv2.resize(flow_map.astype(background.dtype), (frame_width, frame_height))
            if merge == 2:
                img = np.hstack((mask, background))
            else:
                maskgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                ret, maskmask = cv2.threshold(maskgray, 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(maskmask)
                background_bg = cv2.bitwise_and(background, background, mask = mask_inv)
                mask_fg = cv2.bitwise_and(mask, mask, mask = maskmask)
                img = cv2.add(background_bg, mask_fg)

            cv2.imwrite(vediolib + 'FinalPlot_'  + str(sec) + 's/' + each_video_name + '_finalplot.jpg', img)
            cap.release()
            #out.release()

def azimuthAngle(x1, y1, dx, dy):
    angle = 0.0;
    x2 = x1 + dx
    y2 = y1 + dy
    if  x2 == x1:
        angle = math.pi / 2.0
        if  y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * math.pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = math.atan(dx / dy)
    elif  x2 > x1 and  y2 < y1 :
        angle = math.pi / 2 + math.atan(-dy / dx)
    elif  x2 < x1 and y2 < y1 :
        angle = math.pi + math.atan(dx / dy)
    elif  x2 < x1 and y2 > y1 :
        angle = 3.0 * math.pi / 2.0 + math.atan(dy / -dx)
    return (angle * 180 / math.pi)


def find_max_dis(cal_points_list, CoLine_list):
    index = 0
    for i in range(len(cal_points_list) - 1, -1, -1):
        #print(cal_points_list[i])
        if cal_points_list[i] in CoLine_list:
            index = i
            break
    return index

def find_points(ends):
    #print(ends)
    d0, d1 = np.diff(ends, axis=0)[0]
    if np.abs(d0) > np.abs(d1): 
        return np.c_[np.arange(ends[0, 0], ends[1,0] + np.sign(d0), np.sign(d0), dtype=np.int32),
                     np.arange(ends[0, 1] * np.abs(d0) + np.abs(d0)//2,
                               ends[0, 1] * np.abs(d0) + np.abs(d0)//2 + (np.abs(d0)+1) * d1, d1, dtype=np.int32) // np.abs(d0)]
    else:
        return np.c_[np.arange(ends[0, 0] * np.abs(d1) + np.abs(d1)//2,
                               ends[0, 0] * np.abs(d1) + np.abs(d1)//2 + (np.abs(d1)+1) * d0, d0, dtype=np.int32) // np.abs(d1),
                     np.arange(ends[0, 1], ends[1,1] + np.sign(d1), np.sign(d1), dtype=np.int32)]

def find_pres_point(SkyLine_list, mid_point):
    dis_min = 10000
    get_i = 0
    get_j = 0
    for item in SkyLine_list:
        dis_get = int(sqrt(abs(item[0] - mid_point[0]) ** 2 + abs(item[1] - mid_point[1]) ** 2))
        if dis_get < dis_min:
            dis_min = dis_get
            get_i = item[0]
            get_j = item[1]
    return [get_i, get_j]

def find_skyline_mid(img):
    SkyLine_list = []
    CoLine_list = []
    CoLine = cv2.medianBlur(img, 51)
    CoLine_smooth = cv2.GaussianBlur(CoLine, (7, 7), 0)
    CoLine_l = cv2.Canny(CoLine_smooth, 100, 150)
    sky_s = int(CoLine.shape[0] / 3)
    skyline_flag = 1
    for j in range(CoLine.shape[1]):
        flag = -1
        len_l = 0
        sky_flag = 0
        for i in range(CoLine.shape[0]):
            if i < sky_s:
                if CoLine_l[i][j] == 255:
                    SkyLine_list.append([i, j])
                    sky_flag = 1
                CoLine_l[i][j] = 0
            elif CoLine_l[i][j] == 255:
                flag = 1
                CoLine_list.append([i, j])
        if flag == -1:
            CoLine_list.append([CoLine.shape[0] - 1, j])
        if sky_flag == 0:
            skyline_flag = 0

    if skyline_flag == 0:
        SkyLine_list = []
    return CoLine_list, SkyLine_list


def findvideoname(video_dir, framename, vide_type):
    videos = os.listdir(video_dir)
    videos = filter(lambda x: x.endswith(vide_type), videos)
    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        if each_video_name == framename[0:len(each_video_name)]:
            return each_video_name

def use_mask(u, v, mask_value):
    newu = np.zeros(u.shape)
    newv = np.zeros(v.shape)
    for y in range(u.shape[1]):
        for x in range(u.shape[0]):
            if mask_value[x, y] == 0:
                newu[x, y] = u[x, y]
                newv[x, y] = v[x, y]
    return newu, newv

def motionmask(frame, u, v, videoname):
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
       
    grandmaskframe = cv2.resize(cv2.imread('./New_Rip_Mask/' + videoname + '_mask.png'), (mask.shape[1], mask.shape[0]))
    
    grandmask = 255 - cv2.cvtColor(grandmaskframe, cv2.COLOR_BGR2GRAY)
    
    mask = cv2.add(mask, mask2) 

    p_count = np.zeros(mask.shape)
    for y in range(u.shape[1]):
        for x in range(u.shape[0]):
            if mask[x, y] != 0:
                p_count = 1

    mask = cv2.add(mask, grandmask)
    
    for y in range(u.shape[1]):
        for x in range(u.shape[0]):
            if mask[x, y] == 0:
                newu[x, y] = u[x, y]
                newv[x, y] = v[x, y]
    
    return newu, newv, p_count

def heatmask(frame, heatmap):
    newheat = np.zeros(heatmap.shape)
    newframe = cv2.inRange(frame , [0, 0, 0], [200, 200, 200])
    hsv = cv2.cvtColor(newframe, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0, 0, 50])
    upper_blue = np.array([50, 50, 225])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
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
    sec = 20      #Seconds of Demo Video
    core = 30      #Number of cores for processing.
    frame_num = 24 #Number of frames picked once.
    gaptime = 0   #Seconds of Gap Time
    merge = 2     #"2" is merge mode, "1" is normal mode.
    #vediolib = './New_YouTube_Video/' #Video Store Address
    #vediolib = './OptOffTest/'
    #vediolib = './New_Rip_Video/'
    #vediolib = './PalmBeachVideo/'
    vediolib = './DJI_Video/'
    rate = 60
    mode = 'heatmap'
    #mode = 'dye'
    #mode = 'demo'
    vide_type = 'mp4'
    #vide_type = 'MOV'
    gen_vedio((sec + gaptime), core, rate*2, vediolib, mode, gaptime, merge, vide_type, rate)
