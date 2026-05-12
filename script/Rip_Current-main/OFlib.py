import numpy as np
import cv2
import math
import zipfile
import os
import HighOrder
import colorsys
import pywt
import HornSchunck

def compute_flow_map(u, v, gran=20):
    assert u.shape == v.shape
    flow_map = np.zeros((u.shape[0], u.shape[1], 3))
    dxmax = 0
    dymax = 0
    lengthmax = 0
    for y in range(flow_map.shape[0]):
        for x in range(flow_map.shape[1]):
            if y % (gran/5) == 0 and x % (gran/5) == 0:
                dx = 5 * (u[y, x])
                dy = 5 * (v[y, x])
                length = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
                #dxmax = dxmax + dx
                #dymax = dymax + dy
                #lengthmax = lengthmax + length
                if length > lengthmax and length < 20:
                    dxmax = dx
                    dymax = dy
                    lengthmax = length
            if y % gran == 0 and x % gran == 0:
                #dx = 10 * (u[y, x])
                #dy = 10 * (v[y, x])
                #dxmax = float(dxmax / 4)
                #dymax = float(dymax / 4)
                #lengthmax = float(lengthmax / 4)
                scalar = 2
                if (((abs(dxmax) > 0) or (abs(dymax) > 0))):
                    try:
                        #cv2.arrowedLine(flow_map, (x, y), ((x + scalar * (dx)), (y + scalar * (dy))), 255, 1)
                        cv2.arrowedLine(flow_map, (x, y), (int(x + scalar * (dxmax)), int(y + scalar * (dymax))), (255, 255, 255), 2, tipLength = 0.5)
                        #if(dx < 0):
                        #    print("dx is negative at (%s, %s). Phase: %s" % (y, x, float(math.atan(dy/dx))))
                        #if(dy < 0):
                        #    print("dy is negative at (%s, %s. Phase: %s" % (y, x, float(math.atan(dy/dx))))
                    except:
                        continue
                dx = 0
                dy = 0
                length = 0
                dxmax = 0
                dymax = 0
                lengthmax = 0
    return flow_map

def digitaldye(u, v, dye, w, l):
    assert u.shape == v.shape
    #i = 0
    newdye = []
    #newdye = np.zeros(dye.shape)
    flow_map = np.zeros((l, w, 3))
    #for y in range(l - 10):
    #    for x in range(w - 10):
    #        if dye[x][y] != 0:
    #            dx = 10 * u[y, x]
    #            dy = 10 * v[y, x]
    #            if dx > 50: dx = 0
    #            if dy > 50: dy = 0
    #            if (dx > 0) or (dy > 0):
    #                newx = x + dx
    #                newy = y + dy
    #                if (newx > (w - 10)): newx = (w - 10)
    #                if (newx < 10): newx = 10
    #                if (newy > (l - 10)): newy = (l - 10)
    #                if (newy < 10): newy = 10 
    #                newcolor = (dye[x][y] + dye[newx][newy]) / 2 + 0.2
    #                dye[x][y] = newcolor
    #                dye[newx][newy] = newcolor
    #for y in range(l - 10):
    #    for x in range(w - 10):
    #        if dye[x][y] != 0:
    #            cv2.circle(flow_map, (x, y), 3, (255 * dye[x][y]), -1)  
    for eachdye in dye:
        dx = 10 * u[eachdye[0], eachdye[1]]
        dy = 10 * v[eachdye[0], eachdye[1]]
        x = int(eachdye[1] + dx)
        y = int(eachdye[0] + dy)
        if (x > (w - 10)): x = (w - 10)
        if (x < 10): x = 10
        if (y > (l - 10)): y = (l - 10)
        if (y < 10): y = 10
        #if (dy, dx) != (0, 0):
        #    newdye.append([y, x, int(eachdye[2] / 2 + 0.5)])
        #    if not([y, x, eachdye[2]] in newdye): newdye.append([y, x, eachdye[2]])
        #else: 
        #    if not([y, x, int(eachdye[2] / 2 + 0.5)] in newdye): 
        #        newdye.append([y, x, int(eachdye[2] / 2 + 0.5)])
        #cv2.circle(flow_map, (x, y), 3, (255 * eachdye[2]), -1)
        #cv2.circle(flow_map, (x, y), 3, (255 - (i * 5), 255 - (i * 5), 255 - (i * 5)), -1)
        #newdye.append([y, x, 1])
        #if not([y, x] in dye) and (dx + dy) < 50: newdye.append([y, x])
        #i = i + 1
        #print(y, x, eachdye[2])
        if (dx + dy) < 50: newdye.append([y, x, eachdye[2]])
        #cv2.circle(flow_map, (x, y), 3, (eachdye[2][0], eachdye[2][1], eachdye[2][2]), -1)
    #print(newdye)
    #newdye = newdye + dye
    for eachdye in newdye:
        #cv2.circle(flow_map, (eachdye[1], eachdye[0]), 3, (255, 255, 255), -1)
        cv2.circle(flow_map, (eachdye[1], eachdye[0]), 5, (20, 20, 20), -1)
        #print(eachdye[2])
        colorrgb = colorsys.hsv_to_rgb(eachdye[2], 1, 1)
        colorget = (colorrgb[0] * 255, colorrgb[1] * 255, colorrgb[2] * 255)
        cv2.circle(flow_map, (eachdye[1], eachdye[0]), 3, colorget, -1)
    return flow_map, newdye

def check_angle(adj_off_angle, dangle, range = 90):
    if (adj_off_angle >= range) and (adj_off_angle <= (360 - range)):
        return abs(adj_off_angle - dangle) < range
    elif adj_off_angle < range:
        if abs(dangle - adj_off_angle) < range:
            return True
        elif (adj_off_angle + 360 - dangle) < range:
            return True
        else:
            return False
    elif adj_off_angle > (360 - range):
        if abs(dangle - adj_off_angle) < range:
            return True
        elif (dangle + 360 - adj_off_angle) < range: 
            return True
        else:
            return False
    else:
        return False

def heatmap(u, v, off_u, off_v, heat, mode, perspective_point=None, maxvalue = 0):
    assert u.shape == v.shape
    if perspective_point is None:
        perspective_point = [0, 0]
    heatmap = np.zeros((v.shape[0], v.shape[1], 3))
    p_count = np.zeros(heat.shape)
    if mode == 1:
        return heatmap, heatmap
    #maindx = 0
    #maindy = 0
    #for y in range(heatmap.shape[0]):
    #    for x in range(heatmap.shape[1]):
    #        dx = float((u[y, x]))
    #        dy = float((v[y, x]))
    #        if (dx != 0 or dy != 0) and (dx + dy) < 10:
    #            maindx += dx
    #            maindy += dy
    #mainangle = azimuthAngle(0, 0, maindx, maindy)
    #maxcolor = max(heat.flatten())
    #print(maxcolor)
    for y in range(2, heatmap.shape[0] - 2):
        for x in range(2, heatmap.shape[1] - 2):
            dx = float((u[y, x]))
            dy = float((v[y, x]))
            off_dx = float((off_u[y, x]))
            off_dy = float((off_v[y, x]))
            if (dx != 0 or dy != 0): #and (dx + dy) < 10:
                if perspective_point != [0, 0] and y != perspective_point[0] and x != perspective_point[1]:
                    ratio = (abs(y - perspective_point[0]) / abs(heatmap.shape[0] - perspective_point[0])) ** 2
                    dx_per = abs(x - perspective_point[1])
                    dy_per = y - perspective_point[0]
                    per_value = math.sqrt(dx_per ** 2 + dy_per ** 2)
                    if per_value != 0:
                    
                        opt_value = math.sqrt(dx ** 2 + dy ** 2)
                        dx_resize = dx / (opt_value / per_value)
                        dy_resize = dy / (opt_value / per_value)
                        dx = dx_resize * ratio + dx_per * (1 - ratio)
                        dy = dy_resize * ratio + dy_per * (1 - ratio)
                dangle = azimuthAngle(x, y, dx, dy)
                off_angle = azimuthAngle(x, y, off_dx, off_dy)
                #adj_off_angle = off_angle - pers_angle
                #if adj_off_angle < 0:
                #    adj_off_angle += 360
                #print(pers_angle, dangle, off_angle)
                if abs(off_angle - dangle) < 90:
                #if check_angle(adj_off_angle, dangle, 90): #Off Angle
                    heat[x][y] += 1
                #if (abs(off_angle - dangle) < 90) != (check_angle(off_angle, dangle, 90)):
                #    print(y, x, off_angle, dangle, pers_angle)
                #elif x > (heatmap.shape[1] / 2) and (abs(off_angle - dangle - abs(d_angle + pers_angle)) < 90):

    if maxvalue == 0:
        maxcolor = max(heat.flatten())
    else:
        maxcolor = maxvalue * 0.4
    if mode != 2:
        for y in range(0, (heatmap.shape[0])):
            for x in range(heatmap.shape[1]):
                if heat[x][y] > 0:
                    color = heat[x][y] / maxcolor
                    if color > 1: color = 1
                    colorrgb = colorsys.hsv_to_rgb(color, 1, 1)
                    colorget = (colorrgb[0] * 255, colorrgb[1] * 255, colorrgb[2] * 255)
                    cv2.circle(heatmap, (x, y), 3, colorget, -1)
                elif heat[x][y] < 0:
                    heat[x][y] = 0

        return heatmap, heat

    return heat

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

def higher_order(previous, current, mode = 'LK'):
    if hasattr(previous, 'shape') and hasattr(current, 'shape'):
        assert previous.shape == current.shape
        levels = 3
        wav = pywt.Wavelet('db6')
        hor = HighOrder.HighOrderRegularizerConv(wav)
        C1 = pywt.wavedec2(previous, hor.wav, level=levels, mode=hor.mode)
        C2 = pywt.wavedec2(current, hor.wav, level=levels, mode=hor.mode)
        l2norm_hor, (grad1, grad2) = hor.evaluate(C1, C2, 'l2norm')
        #hornschunck_hor, (grad1, grad2) = hor.evaluate(C1, C2, 'hornschunck')
        #op_flow_x, op_flow_y = lucas_kanade(C1, C2)
        if mode == 'LK':
            op_flow_x, op_flow_y = lucas_kanade(grad1, grad2)
        if mode == 'HS':
            op_flow_x, op_flow_y = HornSchunck.HS(grad1, grad2)
        return op_flow_x, op_flow_y
    return np.zeros([10, 10]), np.zeros([10, 10])

def lucas_kanade(previous, current, win=3):
    if hasattr(previous, 'shape') and hasattr(current, 'shape'):
        assert previous.shape == current.shape
        I_x = np.zeros(previous.shape)
        I_y = np.zeros(previous.shape)
        I_t = np.zeros(previous.shape)
        I_x[1:-1, 1:-1] = (previous[1:-1, 2:] - previous[1:-1, :-2]) / 2
        I_y[1:-1, 1:-1] = (previous[2:, 1:-1] - previous[:-2, 1:-1]) / 2
        I_t[1:-1, 1:-1] = previous[1:-1, 1:-1] - current[1:-1, 1:-1]
        params = np.zeros(previous.shape + (5,)) #Ix2, Iy2, Ixy, Ixt, Iyt
        params[..., 0] = I_x * I_x # I_x2
        params[..., 1] = I_y * I_y # I_y2
        params[..., 2] = I_x * I_y # I_xy
        params[..., 3] = I_x * I_t # I_xt
        params[..., 4] = I_y * I_t # I_yt
        del I_x, I_y, I_t
        cum_params = np.cumsum(np.cumsum(params, axis=0), axis=1)
        del params
        win_params = (cum_params[2 * win + 1:, 2 * win + 1:] -
                     cum_params[2 * win + 1:, :-1 - 2 * win] -
                     cum_params[:-1 - 2 * win, 2 * win + 1:] +
                     cum_params[:-1 - 2 * win, :-1 - 2 * win])
        del cum_params
        det = win_params[...,0] * win_params[..., 1] - win_params[..., 2] **2
        op_flow_x = np.where(det != 0, (win_params[..., 1] * win_params[..., 3] - win_params[..., 2] * win_params[..., 4]) / det, 0)
        op_flow_y = np.where(det != 0, (win_params[..., 0] * win_params[..., 4] - win_params[..., 2] * win_params[..., 3]) / det, 0)
        return op_flow_x, op_flow_y
    return np.zeros([10,10]), np.zeros([10,10])


def l2norm(U1, U2):
    return 0.5 * np.sum(U1 ** 2 + U2 ** 2)

def hornschunck(U1, U2):
    result = 0.
    g1, g2 = np.gradient(U1)
    result += np.sum(g1 ** 2 + g2 ** 2)
    g1, g2 = np.gradient(U2)
    result += np.sum(g1 ** 2 + g2 ** 2)
    return 0.5 * result

def addfile(zipfilename, dirname):
    if os.path.isfile(dirname):
        with zipfile.ZipFile(zipfilename, 'a', zipfile.ZIP_DEFLATED) as z:
            z.write(dirname)
    else:
        with zipfile.ZipFile(zipfilename, 'a', zipfile.ZIP_DEFLATED) as z:
            for root, dirs, files in os.walk(dirname):
                for single_file in files:
                    if single_file != zipfilename:
                        filepath = os.path.join(root, single_file)
                        z.write(filepath)

def un_zip(file_name):
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(file_name + "_files"):
        pass
    else:
        os.mkdir(file_name + "_files")
    for names in zip_file.namelist():
        zip_file.extract(names, file_name + "_files/")
    zip_file.close() 
