import os

path = '/home/djjin/Test/'
change_name ='traffic_test_'

def changeName(path, cName):
    i = 1
    file_list = os.listdir(path)
    sorted_file_list = sorted(file_list)
    for filename in sorted_file_list:
        filtered_file = filename.replace('pylon_camera_node-','').replace('.jpg','').split('-')
        result_file = ''.join(filtered_file)
        print(path + filename, '=>', path+str(cName)+str(i)+'.jpg')
        os.rename(path+filename, path+str(cName)+str(i)+'.jpg')
        i += 1

changeName(path, change_name)

