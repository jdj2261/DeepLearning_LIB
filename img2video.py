import cv2
import os
import natsort # pip3 install natsort
from tqdm import tqdm # pip3 install tqdm

image_folder = '/home/djjin/Test'
video_name = 'video.avi'
frame = 20

def img2video(img_path, video_name, frame):

    file_list = os.listdir(img_path)
    natsorted_test = natsort.natsorted(file_list,reverse=False)

    images = [img for img in natsorted_test if img.endswith(".jpg")]
    ret = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = ret.shape
    video = cv2.VideoWriter(video_name, 0, frame, (width,height))

    tot_sum = 0
    for i, image in enumerate(tqdm(images)):
        video.write(cv2.imread(os.path.join(image_folder, image)))
        tot_sum += i
    cv2.destroyAllWindows()
    video.release()

img2video(image_folder, video_name, frame)