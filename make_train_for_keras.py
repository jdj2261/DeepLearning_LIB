import os
import natsort
import argparse

def make_train_file(*input):

    str="""
    example
    img_path = '/home/djjin/Test/merge/labels_class_5/'
    """
    if len(input) != 1:
        print(str)
        return

    path = input[0] + "/"

    file_list = os.listdir(path)
    natsorted_files = natsort.natsorted(file_list, reverse=False)
    result_list = []
    red_count = 0
    green_count = 0

    for file in natsorted_files:
        if file.endswith(".txt"):
            with open(path+file, "r") as f:
                while True:
                    data = f.readline().strip('\n')
                    # 내용이 아무것도 없는 경우 다음 파일로 넘어가기
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
                        result_list.append(file.replace('txt','jpg ') + ','.join(data).rstrip())
    print("red: {}, green: {}, total : {}".format(red_count, green_count, red_count + green_count))

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