#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created Date: March 25. 2021
Copyright: UNMANNED SOLUTION
Author: Dae Jong Jin 
Description: coco class name change ( traffic light -> traffic_light )

@example
python3 change_class_name.py --change_class_name $(desired_directory_path) 
'''

import os
import natsort
import argparse

def change_class_name(*input):

    path = input[0] + "/"

    file_list = os.listdir(path)
    natsorted_files = natsort.natsorted(file_list, reverse=False)

    for file in natsorted_files:
        if file.endswith(".txt"):
            with open(path+file, "r") as f:
                lines = f.readlines()

            with open(path+file, "w") as f:
                for line in lines:
                    data = line.split()
                    if not data:
                        continue
                    # 이 부분을 수정하여 class name을 바꾸면 된다.
                    if data[0] == "traffic":
                        data[0] = "traffic_light"
                        data.remove(data[1])
                    print(' '.join(data) + '\n')
                    f.write(' '.join(data) + '\n')
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="change_class_name")
    parser.add_argument(
        '-cn', '--change_class_name', type=str, required=False, nargs='+',
        help='--change_class_name input_path'
    )

    args = parser.parse_args()
    input_list = args.change_class_name
    change_class_name(*input_list)


# make_train_file("/home/djjin/Test/merge/labels_class_5/")