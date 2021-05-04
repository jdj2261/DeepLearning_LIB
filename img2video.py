#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created Date: March 25. 2021
Author: Dae Jong Jin 
Description: Convert img files to video file

@example
python3 img2video.py --img2video $(image directory_path) $(video output_name) $(frame_rate)
python3 img2video.py --img2video ~/Test/test test_output.mp4 30
'''

import cv2
import os
import natsort # pip3 install natsort
import argparse
from tqdm import tqdm # pip3 install tqdm

image_folder = '/home/djjin/Mygit/My_Python_LIB/ETRI/img_test'
video_name = 'video3.avi'
frame = 1

def img2video(*input):
    str="""
    example
    input_file = '/home/djjin/Mygit/My_Python_LIB/ETRI/img_test'
    video_name = 'video3.avi'
    frame = 30
    """
    if len(input) != 3:
        print(str)
        return

    img_path    = input[0] 
    video_name  = input[1] 
    frame       = int(input[2])

    file_list = os.listdir(img_path)
    natsorted_test = natsort.natsorted(file_list,reverse=False)

    images = [img for img in natsorted_test if img.endswith(".jpg")]
    ret = cv2.imread(os.path.join(img_path, images[0]))
    height, width, layers = ret.shape
    video = cv2.VideoWriter(video_name, 0, frame, (width,height))

    tot_sum = 0
    for i, image in enumerate(tqdm(images)):
        video.write(cv2.imread(os.path.join(img_path, image)))
        tot_sum += i
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '-i2v', '--img2video', type=str, required=False, nargs='+',
        help='--img2video image_path output_name frame_rate'
    )

    args = parser.parse_args()
    input_list = args.img2video
    img2video(*input_list)
