import os
import shutil
import natsort
path = '/home/djjin/Test/green/'
change_name ='traffic_test_'

def changeName(path, cName):
    i = 1
    file_list = os.listdir(path)
    natsorted_test = natsort.natsorted(file_list,reverse=False)
    print(len(natsorted_test))

    for filename in natsorted_test:
        if filename.endswith(".jpg"):
            filtered_file = filename.replace('pylon_camera_node-','').replace('.jpg','').split('-')
            result_file = ''.join(filtered_file)
            print(path + filename, '=>', path+str(cName)+str(i)+'.jpg')
            os.rename(path+filename, path+str(cName)+str(i).zfill(5)+'.jpg')
            i += 1
        # else:
        #     shutil.rmtree(path+filename)
        #     print("{}폴더가 삭제되었습니다.".format(filename))
changeName(path, change_name)

