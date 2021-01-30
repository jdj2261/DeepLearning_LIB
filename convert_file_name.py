import os
import shutil
import natsort
import argparse

# # path = '/home/djjin/Mygit/My_Python_LIB/ETRI/'
# change_name ='traffic_test_'

def changeName(path, beforeName, afteName, file_type):
    cName_cnt = 0
    file_list = os.listdir(path)
    natsorted_files = natsort.natsorted(file_list,reverse=False)

    for filename in natsorted_files:
        if filename.endswith(file_type):
            cName_cnt += 1
            print(filename, '=>', str(afteName)+str(cName_cnt)+file_type)
            os.rename(path+filename, path+str(afteName)+str(cName_cnt).zfill(6)+file_type)

        else:
            print("{} 해당하는 타입의 파일이 아닙니다.".format(filename))
        #     shutil.rmtree(path+filename)
        #     print("{}폴더가 삭제되었습니다.".format(filename))
    print("{}개의 파일 중 {}개의 파일이 변경되었습니다.".format(len(natsorted_files),cName_cnt))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '-ch','--change', type=str, required=True, nargs='+',
        help='--change-name file_path change_name'
    )
    parser.add_argument(
        '--before', type=str, required=True,
        help='Current Name'
    )
    parser.add_argument(
        '--after', type=str, required=True,
        help='Desired Name'
    )
    parser.add_argument(
        '--type', type=str, required=True,
        help='--type jpg, txt, png ...'
    )

    FLAGS = parser.parse_args()
    path = ''.join(FLAGS.change)
    before_file_name = ''.join(FLAGS.before)
    after_file_name = ''.join(FLAGS.after)
    input_type = '.'+''.join(FLAGS.type)
    changeName(path, before_file_name, after_file_name, input_type)