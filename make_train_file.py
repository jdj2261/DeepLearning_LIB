import glob
import random

"""
1. seperate train.txt , valid.txt -> ok
2. change file path -> ok
"""
def file_path_save(path, percent):

    percent = percent / 100
    if percent == 1:
        files = sorted(glob.glob(path+'img/*.jpg'))
    else:
        files = glob.glob(path+'img/*.jpg')
        random.shuffle(files)

    train_files = [train_file for train_file in files[:int(len(files) * percent)]]
    train_files = sorted(train_files)

    valid_files = [valid_file for valid_file in files[int(len(files) * percent):]]
    valid_files = sorted(valid_files)

    with open(path+"test/train.txt",'w') as train_file:
        for file in train_files:
            filterd_file = file.replace(directory_path,'data/')
            train_file.write(filterd_file + "\n")

    with open(path+"test/valid.txt",'w') as valid_file:
        for file in valid_files:
            filterd_file = file.replace(directory_path,'data/')
            valid_file.write(filterd_file + "\n")   

directory_path = "/home/djjin/darknet/data/"
percent = 80
file_path_save(directory_path, percent)

# vim command
#  %s/x64\/Release\///i 