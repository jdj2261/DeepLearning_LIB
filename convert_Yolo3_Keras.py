import os
import shutil as sh
import natsort

path = '/home/djjin/Mygit/My_Python_LIB/ETRI/labels_class_5/'
change_name ='ETRI_traffic_'

# To do 
# 폴더 내에 jpg이면 jpg로, txt이면 txt로 변환하기

def changeName(path, cName):
    i = 1
    file_list = os.listdir(path)
    natsorted_files = natsort.natsorted(file_list,reverse=False)
    print(len(natsorted_files))

    for filename in natsorted_files:
        if filename.endswith(".txt"):
            filtered_file = filename.replace('fc2_save_2017-01-17-','').replace('.txt','').split('-')
            result_file = ''.join(filtered_file)
            print(path + filename, '=>', path+str(cName)+str(i)+'.txt')
            os.rename(path+filename, path+str(cName)+str(i).zfill(6)+'.txt')
            i += 1

changeName(path, change_name)

def make_train_file(path):
    file_list = os.listdir(path)
    natsorted_files = natsort.natsorted(file_list, reverse=False)
    # print(natsorted_files)
    result_list = []
    red_count = 0
    green_count = 0
    for i, file in enumerate(natsorted_files):
        with open(path+file, "r") as f:
            test = str(file)
            while True:
                data = f.readline().strip('\n')
                if data == '':
                    break
                else:
                    data = data.split('\t')
                    print(file, data)
                    class_data = data[4]
                    if class_data == '1301' or class_data == '1401':
                        data[4] = '1' #'traffic_red'
                        red_count += 1
                    elif class_data == '1300' or class_data == '1400':
                        data[4] = '0' #'traffic_green'
                        green_count += 1
                    else:
                        continue
                    result_list.append(test.replace('txt','jpg ') + ','.join(data).rstrip())
    print("red: {}, green: {}".format(red_count, green_count))


    with open(path+'../train.txt', 'w') as train_file:
        for file in result_list:
            train_file.write(file + "\n") 

def copy_img(path):
    train_text_list = []
    with open(path+'../train.txt', 'r') as train_file:
        for text in train_file:
            list = text.split(' ')
            train_text_list.append(list[0])
    # print(train_text_list)

    img_path = '/home/djjin/Mygit/My_Python_LIB/ETRI/JPEGImages_mosaic'
    copy_path = '/home/djjin/Mygit/My_Python_LIB/ETRI/img_test'
    img_file_list = os.listdir(img_path)
    natsorted_files = natsort.natsorted(img_file_list,reverse=False)

    for file in natsorted_files:
        if file in train_text_list:
            print(file)
            sh.copy(img_path+file, copy_path)

# current_path = os.getcwd()
# os.chdir(current_path + "/ETRI")
# print("after: %s"%os.getcwd())
# after_path = os.getcwd()
# print(os.listdir(os.getcwd()))

# path = "./python/test"
# if not os.path.isdir(path):                                                           
#     os.mkdir(path)

# 현재 위치 얻기
# current_path = os.getcwd()
# print(current_path)

