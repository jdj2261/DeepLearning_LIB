#ffmpeg -i input.mp4 -ss 00:00:10 -to 00:00:20 -c copy output.mp4
import os
import argparse

def crop_video(*input):
    str="""
    example
    input_file = '/home/djjin/Videos/dwelling.mp4'
    start_time = '00:00:00'
    end_time = '00:03:10'
    output_file = '/home/djjin/Videos/output1.mp4'
        """
    if len(input) != 4:
        print(str)
        return

    current_path = os.getcwd()
    video_path = input[0]
    start_time = input[1]
    end_time   = input[2]
    output     = input[3]
    cmd = "ffmpeg -i " + video_path + " -ss " + start_time + " -to " + end_time + " -c copy " + current_path +"/"+output
    os.system(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '-cr', '--crop', type=str, required=False, nargs='+',
        help='--crop video_path start_time(hour:min:sec) end_time((hour:min:sec)) output_name'
    )

    FLAGS = parser.parse_args()
    input_list = FLAGS.crop
    crop_video(*input_list)

    # crop_video()
# To do
#  ffmpeg -i in.mp4 -vf "scale=1280x720" out.mp4