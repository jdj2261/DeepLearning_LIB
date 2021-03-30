# import os
# import glob
# import natsort

# directory_path = "/home/djjin/test/"

# 현재 디렉토리 리스트
# file_list = os.listdir(directory_path)

# # txt 확장자를 가진 모든 파일 리스트 
# files = glob.glob(directory_path + '/'+ '*.txt')
# print(files)

# 현재 경로 얻기
# current_path = os.path.abspath(os.path.join(os.getcwd(), os.curdir))
# print(current_path)
# parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
# print(parent_path)



"""
Created Date: March 25. 2021
Copyright: UNMANNED SOLUTION
Author: Dae Jong Jin 
Description: mAP 깃 레포지토리에서 실행될 수 있도록 darknet result.txt 파일 형식을 변환

@example
python3 convert_yolov4_result.py --convert_result ~/test/result.txt 
"""
import argparse
import os

SEPARATOR_KEY = "Enter Image Path:"
REMOVE_KEY = "Detection layer:"
IMG_FORMAT = ".jpg"

def convert_result_txt(*input):
    file_name = input[0]

    path, ext = os.path.splitext(file_name)

    with open(file_name, "r") as f:
        lines = f.readlines()
    with open(path + "_modified" + ext, "w") as f:
        for line in lines:
            data = line.lstrip()
            if SEPARATOR_KEY in data:
                data = line.replace(data,SEPARATOR_KEY)
                print(data)
            else:
                if REMOVE_KEY in data:
                    data = line.replace(data, "")
            f.write(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '-cr', '--convert_result', type=str, required=False, nargs='+',
        help='--convert_result result.txt_path'
    )

    args = parser.parse_args()
    input_list = args.convert_result
    convert_result_txt(*input_list)

# ffmpeg -i video.avi -an -r 5 -y -s 1080x1920 trainimg_%d.jpg
# 1초당 5프레임