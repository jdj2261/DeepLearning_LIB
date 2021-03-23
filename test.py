import os
import glob
# import natsort

directory_path = "/home/djjin/test/"

# 현재 디렉토리 리스트
file_list = os.listdir(directory_path)

# # txt 확장자를 가진 모든 파일 리스트 
# files = glob.glob(directory_path + '/'+ '*.txt')
# print(files)

# 현재 경로 얻기
# current_path = os.path.abspath(os.path.join(os.getcwd(), os.curdir))
# print(current_path)
# parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
# print(parent_path)

SEPARATOR_KEY = "Enter Image :"
IMG_FORMAT = ".jpg"
for file in file_list:
    if file.endswith(".txt"):
        print(file)
        with open(directory_path + file, "r") as f:
            lines = f.readlines()
        
        with open(directory_path + file, "w") as f:
            for line in lines:
                # if image_path = re.search(SEPARATOR_KEY + '(.*)' + IMG_FORMAT, line)
                data = line.strip()
                f.write(data)
