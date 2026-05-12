#################################################
# Anchen Sun
# Sep 17 2021
# Optimized Off-Shore Direction Calcuation
#################################################

import numpy as np
import cv2
import math
import os
import time
import concurrent.futures
from sympy import symbols, Eq, solve, log
from OFlib import azimuthAngle

#import copy

#Find Coastal Line
def DetCoLine(img):
    CoLine = cv2.medianBlur(img, 51)
    CoLine_smooth = cv2.GaussianBlur(CoLine, (7, 7), 0)
    #CoLine = cv2.Canny(CoLine_smooth, 50, 150)
    CoLine_l = cv2.Canny(CoLine_smooth, 100, 150)
    sky_s = int(CoLine.shape[0] / 3)
    for j in range(CoLine.shape[1]):
        #flag = -1
        len_l = 0
        for i in range(CoLine.shape[0]):
            #len_l += 1 
            if i < sky_s:
            #    CoLine[i][j] = 0
                CoLine_l[i][j] = 0
            #elif CoLine[i][j] == 255 and len_l > 3:
            #    flag = flag * -1
            #    len_l = 0
            #if flag == 1:
            #    CoLine[i][j] = 255
            if CoLine_l[i][j] == 255:
                flag = 1
        #if flag == -1:
        CoLine_l[CoLine.shape[0] - 1][j] = 255
    return CoLine_l #, CoLine

#Generate Potential flow
def GenPotFlow(CoLine, CoLine_img, img,  VecFlag = True):
    FlowMatrix_img = CoLine
    PotFlow_v = np.zeros((CoLine.shape[0], CoLine.shape[1]))
    PotFlow_u = np.zeros((CoLine.shape[0], CoLine.shape[1]))
    #CoLine_base = cv2.GaussianBlur(CoLine_img, (21, 21), 0)
    
    FlowMatrix_u = np.zeros(img.shape)
    FlowMatrix_v = np.zeros(img.shape)
    #mask = 255 - img
    '''
    if VecFlag:
        FlowMatrix_u, FlowMatrix_v = Find_Direction_Matrix(FlowMatrix_u, FlowMatrix_v, FlowMatrix_img, mask)
    for sh in range(150, 250, 10):
        PotFlow = cv2.GaussianBlur(CoLine_img, (int(sh * sh / 200) * 2 + 1, int(sh * sh / 100) * 2 + 1), 0)
        #FlowMatrix = FlowMatrix + cv2.Canny(PotFlow + sh, 10 + int(sh / 5), 100)
        if sh < 200:
            new_line = cv2.Canny(cv2.addWeighted(PotFlow, 0.9, CoLine_base, 0.1, 0.0) + sh, 30, 100)
        else:
            new_line =  cv2.Canny(PotFlow + sh, 10 + int(sh / 5), 100)
        FlowMatrix_img = FlowMatrix_img + new_line
        if VecFlag:
            FlowMatrix_u, FlowMatrix_v = Find_Direction_Matrix(FlowMatrix_u, FlowMatrix_v, new_line, mask)        
    '''
    Dis_Matrix = Find_Dis_Matrix(CoLine, img)   
    FlowMatrix_u, FlowMatrix_v = np.gradient(Dis_Matrix) 

    return FlowMatrix_u, FlowMatrix_v, FlowMatrix_img

#Find Distance Matrix based on detected coline and original img
def Find_Dis_Matrix(CoLine, img):
    line_list = []
    Dis_Matrix = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if CoLine[i, j] == 255:
                line_list.append([i, j])
    i_list = []
    j_list = []
    line_list_list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 255:
                i_list.append(i)
                j_list.append(j)
                line_list_list.append(line_list)
                #Dis_Matrix[i, j] = Find_Distance(i, j, line_list)

    with concurrent.futures.ProcessPoolExecutor(24) as executor:
        for dis_get, i, j in executor.map(Find_Distance, i_list, j_list, line_list_list):
            Dis_Matrix[i, j] = dis_get
    return Dis_Matrix

#Find Direction Matrix based on Generated flowimg
def Find_Direction_Matrix(u, v, img, mask):
    u_get = 0
    v_get = 0
    orignal_angle = 0
    changed_angle = 0
    line_list = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 255:
                line_list.append([i, j])

    with concurrent.futures.ProcessPoolExecutor(36) as executor:

        for i in range(img.shape[0]):
            #print(i)
            #for j in range(img.shape[1]):
            i_list = [i] * img.shape[1]
            line_list_list = [line_list] * img.shape[1]
            j = range(img.shape[1])
            j_num = -1
            mask_list = [mask]  * img.shape[1]
            for u_get, v_get in executor.map(Find_Direction, i_list, j, line_list_list, mask_list):
            #print(i, j)
                j_num += 1
                dx = float((u[i, j_num]))
                dy = float((v[i, j_num]))
                if dx == 0 and dy == 0:
                    u[i, j_num], v[i, j_num] = u_get, v_get
                else:
                    #print(dx, dy)
                    orignal_angle = azimuthAngle(i, j_num, dx, dy)
                    changed_angle = azimuthAngle(i, j_num, u_get, v_get)
                    if abs(changed_angle - orignal_angle) < 90:
                        u[i, j_num], v[i, j_num] = u_get, v_get
            '''
            if mask[i, j] == 0:
                if u[i, j] == 0 and v[i, j] == 0:
                    u[i, j], v[i, j] = Find_Direction(i, j, line_list)
                else:
                    orignal_angle = azimuthAngle(i, j, u[i, j], v[i, j])
                    u_get, v_get = Find_Direction(i, j, line_list)
                    changed_angle = azimuthAngle(i, j, u_get, v_get)
                    if abs(changed_angle - orignal_angle) < 90:
                        u[i, j], v[i, j] = u_get, v_get
            '''
    return u, v
     
#Find Distance
def Find_Distance(i, j, line_list):
    dis_min = 10000
    #get_i = 0
    #get_j = 0

    for item in line_list:
        dis_get = int(math.sqrt(abs(item[0] - i) ** 2 + abs(item[1] - j) ** 2))
        if dis_get < dis_min:
            dis_min = dis_get
            #get_i = item[0]
            #get_j = item[1]
    #line_len = len(line_list)
    #with concurrent.futures.ProcessPoolExecutor(32) as executor:
    #    for dis in executor.map(Dis_Get, line_list[:][0], line_list[:][1], [i] * line_len, [j] * line_len):
    #        if dis < dis_min:
    #            dis_min = dis
    return dis_min, i, j

def Dis_Get(x, y, i, j):
    return int(math.sqrt(abs(x - i) ** 2 + abs(y - j) ** 2))

#Find Direction based on Generated flowimg
def Find_Direction(i, j, line_list, mask):
    if mask[i ,j] != 0:
        return 0, 0
    dis_min = 10000
    get_i = 0
    get_j = 0
    for item in line_list:
        dis_get = int(math.sqrt(abs(item[0] - i) ** 2 + abs(item[1] - j) ** 2))
        if dis_get < dis_min:
            dis_min = dis_get
            get_i = item[0]
            get_j = item[1]

    '''
    for k in range(1, int(FlowMatrix_img.shape[0] / 2)):
        if k > dis_min:
           break
        else:
            start_point_i = max(i - k, 0)
            start_point_j = max(j - k, 0)
            end_point_i = min(FlowMatrix_img.shape[0], i + k)
            end_point_j = min(FlowMatrix_img.shape[1], j + k)
    
            for q in range(start_point_i, end_point_i):
                if FlowMatrix_img[q, start_point_j] == 255 and start_point_j != 0:
                    dis_get = int(math.sqrt(abs(q - i) ** 2 + abs(start_point_j - j) ** 2))
                    if dis_get < dis_min:
                        dis_get = dis_min
                        get_i = q
                        get_j = start_point_j
                
                if FlowMatrix_img[q, end_point_j] == 255 and end_point_j!= FlowMatrix_img.shape[1]:
                    dis_get = int(math.sqrt(abs(q - i) ** 2 + abs(end_j - j) ** 2))
                    if dis_get < dis_min:
                        dis_get = dis_min
                        get_i = q
                        get_j = end_point_j

            
            for q in range(start_point_j, end_point_j):
                if FlowMatrix_img[start_point_i, q] == 255 and start_point_i != 0:
                    dis_get = int(math.sqrt(abs(q - j) ** 2 + abs(start_point_i - i) ** 2))
                    if dis_get < dis_min:
                        dis_get = dis_min
                        get_i = start_point_i
                        get_j = q
                
                if FlowMatrix_img[end_point_i, q] == 255 and end_point_i != FlowMatrix_img.shape[0]:
                    dis_get = int(math.sqrt(abs(q - j) ** 2 + abs(end_point_i - i) ** 2))
                    if dis_get < dis_min:
                        dis_get = dis_min
                        get_i = end_point_i
                        get_j = q
    '''  
    dx = get_i - i
    dy = get_j - j
    return dx, dy 

#Read Mask
def ReadMask(videoname):
    imgframe = cv2.imread('./New_Rip_Mask/' + videoname + '_mask.png')
    img = cv2.cvtColor(imgframe, cv2.COLOR_BGR2GRAY)
    return img 

#Calculate Off-Shore Direction Matrix
def OffshoreDir(videoname, superresolution=1, VecFlag = True):
    img = ReadMask(videoname)
    #CoLine, CoLine_img = DetCoLine(img)
    width = int(img.shape[1] * superresolution)
    height = int(img.shape[0] * superresolution)
    dim = (width, height)
    super_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    CoLine = DetCoLine(super_img)
    #cv2.imwrite('CoLine.jpg', CoLine)
    CoLine_img = 0
    u, v, img = GenPotFlow(CoLine, CoLine_img, super_img, VecFlag)
    return u, v, img


#Generate Off Shore Direction Martrix for Lib
def OffshoreDirLib(videolib, superresolution = 1):
    videos = os.listdir(videolib)
    videos = filter(lambda x: x.endswith('MOV'), videos)
    for each_video in videos:
        each_video_name, _ = each_video.split('.')
 
        start = time.time()
        print('Start Process: ', each_video_name)
        
        u, v, img = OffshoreDir(each_video_name, superresolution)
        cv2.imwrite(videolib + each_video_name + '_OffShore_1by1.jpg', img)
        # Bug fix (2026-05-03): was np.int8 — distance-to-coast vectors easily exceed
        # 127 px on a 1920x1080 frame, wrapping to wrong-sign offshore reference for
        # far-from-coast pixels. Use float32. See project_2020_code_audit.md.
        u_save = u.astype(np.float32)
        v_save = v.astype(np.float32)
        np.savez(videolib + each_video_name + '_OffShoreDir_1by1', u_save, v_save)
     
        end = time.time()
        cost = end - start
        m, s = divmod(cost, 60)
        h, m = divmod(m, 60)
        print('The time cost is: ', '%d:%02d:%02d' % (h, m, s))

    return


if __name__ == "__main__":
    #_, _, get = OffshoreDir('Haulover_12_11_2020', False)
    #video_lib = './New_YouTube_Video/'
    video_lib = './PalmBeachVideo/'
    OffshoreDirLib(video_lib)
    #get = OffshoreDir('Grand_New')
    #cv2.imwrite('test.jpg', get) 
