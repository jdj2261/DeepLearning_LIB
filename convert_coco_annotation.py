import os
import natsort
import argparse

def make_train_file(*input):

    help_str="""
    example
    img_path = '/home/djjin/Test/merge/labels_class_5/'
    """
    if len(input) != 1:
        print(help_str)
        return

    path = input[0] + "/"

    file_list = os.listdir(path)
    natsorted_files = natsort.natsorted(file_list, reverse=False)
    result_list = []

    for file in natsorted_files:
        if file.endswith(".txt"):
            with open(path+file, "r") as f:
                lines = f.readlines()

            with open(path+file, "w") as f:
                test = str(file)
                for line in lines:
                    data = line.split()
                    print(data)
                    if data[0] != "None":
                        print(' '.join(data) + '\n')
                        f.write(' '.join(data) + '\n')
                result_list.append(str(test).replace('txt','jpg'))
            
    for file in natsorted_files:
        if file.endswith(".txt"):
            with open(path+file, "r") as f:
                lines = f.readlines()
                if len(lines) == 0:
                    os.remove(path+file)
                    os.remove(path+file.replace('txt','jpg'))

    file_list = os.listdir(path)
    print(len(natsorted_files))
    # print("red: {}, green: {}, total : {}".format(red_count, green_count, red_count + green_count))

    with open(path+'../train.txt', 'w') as train_file:
        for file in result_list:
            train_file.write(file + "\n") 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '-mt', '--make_train', type=str, required=False, nargs='+',
        help='--make_train input_path file_type'
    )

    args = parser.parse_args()
    input_list = args.make_train
    make_train_file(*input_list)


# make_train_file("/home/djjin/Test/merge/labels_class_5/")