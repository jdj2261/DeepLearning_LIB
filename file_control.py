import glob
from os.path import dirname
import shutil as sh
import os
import natsort
import argparse

"""
To do...
1. 파일이 이미 존재 하는 경우
2. 확장자
"""
class file_processing():
    def __init__(self, cur_path, tar_path, file_type):
        self.cur_path = cur_path
        self.tar_path = tar_path
        self.file_type = file_type
        self.file_list = []

    # 디렉토리 확인 
    def find_directories(self):
        cur_file_list = os.listdir(self.cur_path)
        print("current path : {}".format(self.cur_path))

        directory_list = []
        for directory in cur_file_list:
            if os.path.isdir(self.cur_path+'/'+directory):
                print("Directory : {}".format(self.cur_path+'/'+directory))
                directory_list.append(directory)

        if directory_list:
            print('==> There are found {} directories.'.format(len(directory_list)))
        else:
            print('There is no directory...')
        print('=' * 40)

    # 파일 확인
    def find_files(self):

        print(self.file_type)
        file_list = glob.glob(self.cur_path + '*/*'+self.file_type)
        print(self.cur_path + '*/*'+self.file_type)
        natsorted_files = natsort.natsorted(file_list,reverse=False)

        print('>>> file list')
        for file in natsorted_files:
            print(file)
        print('=' * 40)
        print('==> There are found {} files.'.format(len(natsorted_files)))
        print('=' * 40)
        self.file_list = natsorted_files

    # 파일 복사
    def file_copy(self):

        # print(self.file_list)
        for file in self.file_list:
            # file_name = file.split('\\')[-1] # file_name = 파일명.type
            sh.copy(file, self.tar_path)

        tar_file_list = glob.glob(self.tar_path+'*/*' + self.file_type)
        tar_natsorted_files = natsort.natsorted(tar_file_list,reverse=False)
        for cp_file in tar_natsorted_files:
            print("Copied.... >>> {}".format(cp_file))

        print('=' * 40)
        print('==> There are found {} files.'.format(len(tar_natsorted_files)))
        print('=' * 40)
        print('file copy success.')

    # 파일 이동
    def file_move(self):

        for file in self.file_list:
            try:
                sh.move(file, self.tar_path )
            except sh.Error as e:
                print(e)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '-cp', '--copy', type=str, required=False, nargs='+',
        help='--copy current_path desired_path type'
    )

    parser.add_argument(
        '-mv', '--move', type=str, required=False, nargs='+',
        help='--move current_path desired_path type'
    )

    args = parser.parse_args()
    isCopy = False

    if "copy" in args:
        input_list = args.copy
        isCopy = True
    elif "move" in args:
        input_list = args.move

    cur_path = input_list[0]
    tar_path = input_list[1]

    if len(input_list) > 2:
        file_type = input_list[2]
    else:
        file_type = '*'
    file_type = '.' + file_type
    f = file_processing(cur_path=cur_path ,tar_path = tar_path, file_type = file_type)
    f.find_directories()
    f.find_files()
    if isCopy:
        f.file_copy()
    else:
        f.file_move()

