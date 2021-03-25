#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created Date: March 25. 2021
Copyright: UNMANNED SOLUTION
Author: Dae Jong Jin 
Description: Split target txt or jpg file to train.txt and valid.txt

@example
python3 split_file.py --split $(directory path) $(desired directory path) $(Related file type) $(percentage)
python3 split_file.py -sp ~/Mygit/My_Python_LIB/ETRI/img_test test jpg 80
'''
import glob
import random
import argparse
import os

"""
1. seperate train.txt , valid.txt -> ok
2. change file path -> ok
"""
def split_file(*input):
    str="""
    example
    img_path = '/home/djjin/Mygit/My_Python_LIB/ETRI/img_test'
    directry_name = 'test'
    file_type = jpg
    percent = 80
    """
    if len(input) != 4:
        print(str)
        return

    img_path        = input[0] 
    directry_name   = input[1] 
    type            = input[2]
    percent         = int(input[3])

    percent = percent / 100

    if percent == 1:
        files = sorted(glob.glob(img_path + '/*.' + type))
    else:
        files = glob.glob(img_path + '/*.' + type)
        random.shuffle(files)

    train_files = [train_file for train_file in files[:int(len(files) * percent)]]
    train_files = sorted(train_files)

    valid_files = [valid_file for valid_file in files[int(len(files) * percent):]]
    valid_files = sorted(valid_files)

    directory_path = img_path + "/" + directry_name
    print(directory_path)
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    else:
        print("디렉토리가 이미 존재합니다.")
        
    with open(directory_path+"/train.txt",'w') as train_file:
        for file in train_files:
            filterd_file = file.replace(img_path, directry_name)
            print("train.txt >> {}".format(filterd_file))
            train_file.write(filterd_file + "\n")

        print("{} 위치에 {} 파일이 생성되었습니다.".format(directory_path, "train.txt"))
    
    if percent != 1:
        print("=" * 20)
        with open(directory_path+"/valid.txt",'w') as valid_file:
            for file in valid_files:
                filterd_file = file.replace(img_path, directry_name)
                print("valid.txt>> {}".format(filterd_file))
                valid_file.write(filterd_file + "\n")   
            print("{} 위치에 {} 파일이 생성되었습니다.".format(directory_path, "valid.txt"))
    else:
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '-sp', '--split', type=str, required=False, nargs='+',
        help='--split input_path directry_name file_type percentage'
    )

    args = parser.parse_args()
    input_list = args.split
    split_file(*input_list)


# vim command
#  %s/x64\/Release\///i 