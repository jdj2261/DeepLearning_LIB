#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created Date: April 6. 2021
Copyright: UNMANNED SOLUTION
Author: Dae Jong Jin
Description: 에트리 데이터 셋을 Yolo 어노테이션 형태로 변환 
             (디렉토리 안에 이미지와 텍스트 파일이 존재 해야함)

@example
python3 convert_ETRI_to_yolo.py --convert_yolo $(directory path)
python3 convert_ETRI_to_yolo.py --convert_yolo ~/test/
'''

import os
import natsort
import argparse

from PIL import Image

SEARCH_KEY = "_modified"
class ETRI2YOLO:
    def __init__(self, *input) -> None:
        self.path = input[0]
        self.data = ()
        self.class_data = None
    
    def run(self):
        file_list = os.listdir(self.path)
        natsorted_files = natsort.natsorted(file_list, reverse=False)
        self.process_file(natsorted_files)

    def process_file(self, file_list: list) -> None:
        tmp_list = []
        for file in file_list:
            if SEARCH_KEY in file:
                continue

            if file.endswith(".txt"):
                self.txt_path = self.path + file
                self.txt_path, self.ext = os.path.splitext(self.txt_path)
                with open(self.path + file, 'r') as f:
                    lines = f.readlines()
                    data = []
                    for _, line in enumerate(lines):
                        line = line.strip().split('\t')
                        if not line:
                            continue
                        data = line[:4]
                        # print(data)
                        data = list(map(int, data))
                        self.data = data
                        self.class_data = line[4]

                box = (data[0], data[1], data[2], data[3])

                tmp_list.append(box)
                size, box = tmp_list
                print(tmp_list)

                modified_box = self.convert_box(size, box)
                yolo_annotation = self.convert_annotation(modified_box)
                self.write_file(yolo_annotation)
                tmp_list = []

            elif file.endswith(".jpg"):
                im = Image.open(self.path + file)
                w, h = int(im.size[0]), int(im.size[1])
                size = (w, h)
                tmp_list.append(size)

    def convert_box(self, size: tuple, box: tuple) -> tuple:
        dw = 1. / size[0]
        dh = 1. / size[1]
        
        x = (box[1] + box[3])/2.0
        y = (box[0] + box[2])/2.0
        w = (box[2] - box[0])
        h = (box[3] - box[1])

        x *= dw
        y *= dh
        w *= dw
        h *= dh

        return (x, y, w, h)

    def convert_annotation(self, box: tuple) -> list:
        yolo_annotation = list(map(str, box))
        yolo_annotation.append(self.class_data)
        yolo_annotation[0], yolo_annotation[4] = yolo_annotation[4], yolo_annotation[0]
        return yolo_annotation

    def write_file(self, yolo_annotation: list) -> None:
        # print(yolo_annotation)
        with open(self.txt_path + "_modified"+ self.ext,"w") as f:
            print(' '.join(yolo_annotation) + '\n')
            f.write(' '.join(yolo_annotation) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="convert_ETRI_to_yolo")
    parser.add_argument(
        '-cy', '--convert_yolo', type=str, required=False, nargs='+',
        help='--convert_yolo input_path'
    )
    args = parser.parse_args()
    input_list = args.convert_yolo
    etri2yolo = ETRI2YOLO(*input_list)
    etri2yolo.run()