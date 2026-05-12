import numpy as np
import cv2
import matplotlib.pyplot as plt
import zipfile
import os
import concurrent.futures
import time
from OFlib import compute_flow_map, digitaldye, heatmap, un_zip
from OptOffShoreDirection import OffshoreDir

def gen_vedio(sec, core, vediolib, mode, gaptime, merge):
    videos = os.listdir(vediolib)
    videos = filter(lambda x: x.endswith('zip'), videos)
    for each_video in videos:
        print(each_video)
        each_video_name, _ = each_video.split('.')
        with concurrent.futures.ProcessPoolExecutor(core) as executor:
            # cv2.namedWindow("flow map", cv2.WINDOW_NORMAL)
            if os.path.isdir(vediolib + each_video_name):
                pass
            else:
                os.mkdir(vediolib + each_video_name)
            videoname = findvideoname(vediolib, each_video_name)
            #print(each_video_name)
            #print(videoname)
            #print('Test')
            cap = cv2.VideoCapture(vediolib + videoname + '.MOV') #mp4
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
            if mode == 'dye':
                #print(each_video)
                if videoname == 'DJI_0011':
                    xmin = 600
                    xmax = 800
                    ymin = 1600
                    ymax = 1800
                elif videoname == 'Haulover_Miami_FL':
                    xmin = 550
                    xmax = 750
                    ymin = 600
                    ymax = 800
                elif videoname == 'SouthBeachRip':
                    xmin = 500
                    xmax = 700
                    ymin = 800
                    ymax = 1000
                else:
                    xmin = int(frame_height / 2) - 100
                    xmax = int(frame_height / 2) + 100
                    ymin = int(frame_width / 2) - 100
                    ymax = int(frame_width / 2) + 100
                    #print(xmin, xmax)
                dye = []
                # dye = np.zeros([frame_width, frame_height])
                for xx in range(xmin, xmax, 10):
                    col = 0
                    for yy in range(ymin, ymax, 10):
                        # dye[xx][yy] = 1
                        col = col + 1
                        #dye.append([xx, yy])
                        dye.append([xx, yy, (1 / 400 * col)])
                        #dye.append([xx, yy, [0 + col * 10, 255 - col * 10, 700 - xx]])  # [random.randint(0, 200), random.randint(0, 200), random.randint(0, 200)]])

            if mode == 'heatmap':
                heat = np.zeros([frame_width, frame_height])
                offdirname = vediolib + videoname + '_OffShoreDir_1by1.npz'
                if os.path.exists(offdirname):
                    #offdirget = framezip.open(offdirname)
                    offdirectionget = np.load(offdirname)
                    off_u, off_v = offdirectionget["arr_0"], offdirectionget["arr_1"]
                    flow_map = compute_flow_map(off_v, off_u)
                    cv2.imwrite(vediolib + 'offdirflowmap.jpg', flow_map)
                else:
                    off_u, off_v, _ = OffshoreDir(videoname)
            for name in frame_names:
                start = time.time()
                i = i + 1
                # fileget = framezip.extract(name)
                fileget = framezip.open(name)
                arrayget = np.load(fileget)
                u = arrayget["arr_0"]
                v = arrayget["arr_1"]
                # background = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
                background = cap.read()[1]
                # plotframe(u, v, each_video_name, name, i)
                if mode == 'demo':
                    flow_map = compute_flow_map(u, v)
                if mode == 'dye':
                    if (i < gaptime * rate):
                        flow_map, empty = digitaldye(u, v, dye, frame_width, frame_height)
                    else:
                        flow_map, dye = digitaldye(u, v, dye, frame_width, frame_height)
                if mode == 'heatmap':
                    if (i < gaptime * rate):
                        flow_map, empty = heatmap(u, v, off_u, off_v, heat, 1)
                    else:
                        u, v = motionmask(background, u, v, videoname)#, mask_value)
                        flow_map, heat = heatmap(u, v, off_u, off_v, heat, 3)
                        #heat = heatmask(background, heat)
                #print(len(dye))
                # width = background.shape[1] - flow_map.shape[1]
                # height = background.shape[0] - flow_map.shape[0]
                # mask = np.pad(flow_map, ((width,0),(height,0)),'constant',constant_values = (0,0)).astype(background.dtype)
                mask = cv2.resize(flow_map.astype(background.dtype), (frame_width, frame_height))
                # empty = np.zeros_like(mask)
                # img = background
                # for eachdye in dye:
                #    img = cv2.circle(img, (eachdye[0], eachdye[1]), 3, (eachdye[2][0], eachdye[2][1], eachdye[2][2]), -1)
                # RGBTest
                # maskrgb = np.array([mask,mask,mask])
                # maskrgb = maskrgb.swapaxes(0,1)
                # maskrgb = maskrgb.swapaxes(1,2)
                # dst = 255 - maskrgb
                # img = mask & background
                # print(mask.size, background.size)
                # BGTest
                # mask_inv = cv2.bitwise_not(mask)
                # background_bg = cv2.bitwise_and(background, background, mask = mask_inv)
                # mask_bg = cv2.bitwise_and(mask, mask, mask = mask)
                # dst = cv2.add(background_bg, mask_bg)
                # img = dst
                if merge == 2:
                    img = np.hstack((mask, background))
                    #cv2.imwrite(vediolib + each_video_name + '/frame_test' + str(i) + '.jpg', background)
                else:
                    maskgray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    ret, maskmask = cv2.threshold(maskgray, 10, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(maskmask)
                    background_bg = cv2.bitwise_and(background, background, mask = mask_inv)
                    mask_fg = cv2.bitwise_and(mask, mask, mask = maskmask)
                    img = cv2.add(background_bg, mask_fg)

                if i >= (gaptime * rate):
                    cv2.imwrite(vediolib + each_video_name + '/frame' + str(i) + '.jpg', img)
                    #out.write(img)
                if i > sec * rate:
                    cap.release()
                    out.release()
                    # Bug fix (2026-05-03): was np.int8 — wrapped values > 127 to -128,
                    # destroying recall in the most-rip pixels. Use float32 to preserve
                    # the full accumulator range. See project_2020_code_audit.md.
                    heat_save = heat.astype(np.float32)
                    np.savez(vediolib + each_video_name + '_HeatSave', heat_save)
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

def motionmask(frame, u, v, videoname):#, mask_value):
    newu = np.zeros(u.shape)
    newv = np.zeros(v.shape)
    #if mask_value != []:
    #    mask = mask_value
    #else:
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
    
    grandmaskframe = cv2.imread('./New_Rip_Mask/' + videoname + '_mask.png')
    #print(grandmaskframe)
    grandmask = 255 - cv2.cvtColor(grandmaskframe, cv2.COLOR_BGR2GRAY)
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
    return newu, newv #, mask

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
    sec = 20      #Seconds of Demo Video
    core = 8      #Number of cores for processing.
    gaptime = 0   #Seconds of Gap Time
    merge = 2     #"2" is merge mode, "1" is normal mode.
    #vediolib = './New_Rip_Video/' #Video Store Address
    vediolib = './OptOffTest/'
    mode = 'heatmap'
    #mode = 'dye'
    #mode = 'demo'
    gen_vedio((sec + gaptime), core, vediolib, mode, gaptime, merge)
