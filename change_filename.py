#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created Date: March 25. 2021
Copyright: UNMANNED SOLUTION
Author: Dae Jong Jin
Description: Rename all files in the directory

@example
python3 change_filename.py --change $(directory path) --name $(desired_file_name) --type $(file_type)
python3 change_filename.py --change test_dir --name file_name_ --type jpg
'''

import os
import natsort
import argparse

def changeName(path, afterName, file_type):
    startNumber = 0
    cName_cnt = startNumber
    file_list = os.listdir(path)
    natsorted_files = natsort.natsorted(file_list,reverse=False)

    for filename in natsorted_files:
        if filename.endswith(file_type):
            cName_cnt += 1
            print(filename, '=>', str(afterName)+str(cName_cnt)+file_type)
            os.rename(path+filename, path+str(afterName)+str(cName_cnt).zfill(6)+file_type)

        else:
            print("{}은 해당하는 타입의 파일이 아닙니다.".format(filename))
        #     shutil.rmtree(path+filename)
        #     print("{}폴더가 삭제되었습니다.".format(filename))
    print("{}개의 파일 중 {}개의 파일이 변경되었습니다.".format(len(natsorted_files), cName_cnt - startNumber))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="change_filename")
    '''
    Command line options
    '''
    parser.add_argument(
        '-ch','--change', type=str, required=True, nargs='+',
        help='--change file directory'
    )
    parser.add_argument(
        '--name', type=str, required=True,
        help='--name Desired Name'
    )
    parser.add_argument(
        '--type', type=str, required=True,
        help='--type jpg, txt, png ...'
    )

    args = parser.parse_args()
    path = ''.join(args.change) + '/'
    changed_filename = ''.join(args.name)
    input_type = '.'+''.join(args.type)
    changeName(path, changed_filename, input_type)