#ffmpeg -i input.mp4 -ss 00:00:10 -to 00:00:20 -c copy output.mp4
import os

input_file = '/home/djjin/Videos/test.mp4'
start_time = '00:00:00'
end_time = '00:19:10'
output_file = '/home/djjin/Videos/output.mp4'
cmd = "ffmpeg -i " + input_file + " -ss " + start_time + " -to " + end_time + " -c copy " + output_file
print(cmd)
os.system(cmd)