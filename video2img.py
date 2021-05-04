#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created Date: March 25. 2021
Author: Dae Jong Jin 
Description: Convert video file to image files

@example
python3 video2img.py --video2img $(video name) $(image directory path) $(file_type)
python3 video2img.py --video2img ~/Test/test4/output.mp4 ~/Test/test4/img jpg
'''

import cv2
import argparse
from datetime import datetime


# for videoFile in range(len(videoFiles)):
def video2img(*input):
    out_str="""
    example
    video_file = '/home/djjin/Mygit/My_Python_LIB/video.avi'
    save_path = '/home/djjin/Mygit/My_Python_LIB/test'
    file_type = 'jpg'
    """

    if len(input) != 3:
        print(out_str)
        return

    videoFile = input[0] 
    save_path = input[1] + "/"
    file_type = input[2]

    cam = cv2.VideoCapture(videoFile)
    currentFrame = 0

    while(cam.isOpened()):
        now = datetime.now()
        ret, frame = cam.read()
        current_time = str(now.month)+'_'+str(now.day)+'_'+str(now.hour)
        img_name = save_path+current_time+'_'+str(currentFrame).zfill(6) + '.' + file_type
        if ret:
            cv2.imwrite(img_name, frame)
            print(img_name)
            currentFrame += 1
        else:
            break
    cam.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '-v2i', '--video2img', type=str, required=False, nargs='+',
        help='--video2img video_file save_path file_type'
    )

    args = parser.parse_args()
    input_list = args.video2img
    video2img(*input_list)

# ffmpeg -i video.avi -an -r 5 -y -s 1080x1920 trainimg_%d.jpg
# 1초당 5프레임