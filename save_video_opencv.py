# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import time
import os
import datetime

videopath = 'output.avi'

# Checks and deletes the output file
# You cant have a existing file or it will through an error
if os.path.isfile(videopath):
    os.remove(videopath)

cap = cv2.VideoCapture(2)

cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
cv2.resizeWindow('frame', (640, 480))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=cap.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
out = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh))

currentFrame = 0
starttime = time.time()
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        # Handles the mirroring of the current frame
        frame = cv2.flip(frame,1)

        # Saves for video
        out.write(frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # To stop duplicate images
    currentFrame += 1
    totaltime = time.time()-starttime
    print(str(datetime.timedelta(seconds=totaltime)))
# print(currentFrame, "frames", totaltime/currentFrame, "s/frame")
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

