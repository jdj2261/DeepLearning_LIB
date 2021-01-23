import cv2
from tqdm import tqdm # pip3 install tqdm

# for videoFile in range(len(videoFiles)):
def video2img(videoFile, save_path):
    cam = cv2.VideoCapture(videoFile)
    currentFrame = 0
    while(cam.isOpened()):
        ret, frame = cam.read()
        if ret:
            cv2.imwrite(save_path+str(currentFrame) + '.jpg', frame)
            currentFrame += 1
        else:
            break
    cam.release()

if __name__ == '__main__':
    videoFile = '/home/djjin/Mygit/My_Python_LIB/video.avi' 
    save_path = '/home/djjin/Mygit/My_Python_LIB/test/'
    video2img(videoFile, save_path)


# ffmpeg -i video.avi -an -r 5 -y -s 1080x1920 trainimg_%d.jpg
# 1초당 5프레임