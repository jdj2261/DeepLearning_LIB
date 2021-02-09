import glob
from os.path import dirname
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
    percent = 80
    """
    if len(input) != 3:
        print(str)
        return

    img_path    = input[0] 
    directry_name  = input[1] 
    percent       = int(input[2])

    percent = percent / 100
    if percent == 1:
        files = sorted(glob.glob(img_path+'/*.jpg'))
    else:
        files = glob.glob(img_path+'/*.jpg')
        random.shuffle(files)

    train_files = [train_file for train_file in files[:int(len(files) * percent)]]
    train_files = sorted(train_files)

    valid_files = [valid_file for valid_file in files[int(len(files) * percent):]]
    valid_files = sorted(valid_files)

    print(img_path + "/" + directry_name)
    if not os.path.isdir(img_path + "/" + directry_name):
        os.mkdir(img_path + "/" + directry_name)
    else:
        print("디렉토리가 이미 존재합니다.")
        
    with open(img_path + "/" + directry_name+"/train.txt",'w') as train_file:
        for file in train_files:
            filterd_file = file.replace(img_path, directry_name)
            train_file.write(filterd_file + "\n")

    with open(img_path + "/" + directry_name+"/valid.txt",'w') as valid_file:
        for file in valid_files:
            filterd_file = file.replace(img_path, directry_name)
            valid_file.write(filterd_file + "\n")   

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '-sp', '--split', type=str, required=False, nargs='+',
        help='--split input_path directry_name percentage'
    )

    args = parser.parse_args()
    input_list = args.split
    split_file(*input_list)


# vim command
#  %s/x64\/Release\///i 