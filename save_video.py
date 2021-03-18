# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import time
import os
import datetime
import argparse
import sys


def save_video(args: list) -> None:

    if args.webcam:
        # assert args.input is None, "Cannot have both --input and --webcam!"
        # assert args.output is None, "output not yet supported with --webcam!"

        video_number = int(args.webcam)
        cap = cv2.VideoCapture(video_number) # 0, 2, 6
    elif args.video:
        video_path = args.video
        cap = cv2.VideoCapture(video_path) # 0, 2, 6

    if args.output:
        output_name = args.output
    # Checks and deletes the output file
    # You cant have a existing file or it will through an error

    if os.path.isfile(output_name):
        os.remove(output_name)

    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', (640, 480))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret,frame=cap.read()
    time.sleep(1)

    vw = frame.shape[1]
    vh = frame.shape[0]
    print(f"current video frame : {cap.get(cv2.CAP_PROP_FPS)}")
    print ("Video size", vw,vh)
    out = cv2.VideoWriter(output_name, fourcc,30.0,(vw,vh))

    starttime = time.time()
    try:

        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()


            if ret == True:
                # Handles the mirroring of the current frame
                # frame = cv2.flip(frame,1)
                # frame = cv2.flip(frame,0)
                # Saves for video
                out.write(frame)

                # Display the resulting frame
                cv2.imshow('frame',frame)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("save the video..")
                break

            totaltime = time.time()-starttime
            convert_time = str(datetime.timedelta(seconds=totaltime))
            hours = int(float(convert_time.split(":")[0]))

            # if hours == 1:
            #     print("save the video..")
            #     break

            print(f"time : {convert_time}")

    # When everything done, release the capture

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    # 작업들
    except KeyboardInterrupt:
        # Ctrl+C 입력시 예외 발생
        print("save the video..")
        sys.exit() #종료

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="save the video")
    '''
    Command line options
    '''
    parser.add_argument(
        '-save', '--save_video', type=str, required=False, nargs='+',
        help='--save_video output_name'
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--video", help="Path to video file.")
    parser.add_argument("--output",         
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    args = parser.parse_args()

    # input_list = args.img2video
    save_video(args)

# python3 save_video.py --webcam 2 --output /home/djjin/output.mp4