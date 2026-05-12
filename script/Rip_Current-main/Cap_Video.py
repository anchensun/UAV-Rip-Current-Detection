import cv2
import os

def cap_video(vediolib, outputlib):
    videos = os.listdir(vediolib)
    videos = filter(lambda x: x.endswith('mp4'), videos)
    for each_video in videos:
        each_video_name, _ = each_video.split('.')
        cap = cv2.VideoCapture(vediolib + each_video)
        for i in range(10):
            picture = cap.read()[1]
        print(each_video_name + ' Processed')
        cv2.imwrite(outputlib + each_video_name + '.png', picture)

vediolib = './New_Rip_Video/'
outputlib = './CapLib/'
cap_video(vediolib, outputlib) 
