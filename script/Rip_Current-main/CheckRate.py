import cv2
import os

vediolib = 'OptOffTest/'
videos = os.listdir(vediolib)
videos = filter(lambda x: x.endswith('MOV'), videos)
for each_video in videos:
    #videoname = findvideoname(each_video_name, vide_type)
    cap = cv2.VideoCapture(vediolib + each_video)
    rate = int(cap.get(cv2.CAP_PROP_FPS))
    print(each_video, rate)
