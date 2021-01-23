import glob
import shutil as sh

class file_processing():
    def __init__(self, **kwargs):
        if 'path' in kwargs.keys(): 
            self.path = kwargs['path'] 
        else: 
            self.path = None

    def find_file_path_lv1(self):
        ### 대상 파일 경로 확인 ###
        file_list = glob.glob(path + '/*')

        print('>>> file list lv1')
        for file in file_list:
            print(file)
        print('=' * 40)
    
    def find_file_path(self, **kwargs):
        ### 대상 파일 경로 확인 ###
        if 'type' in kwargs.keys():
            type = '.' + kwargs['type']
            file_list = glob.glob(path + '/**/*' + type, recursive=True)
        
        else:
            file_list = glob.glob(path + '/*.jpg')
        
        print('>>> file list')
        for file in file_list:
            print(file)
        print('=' * 40)
        self.file_list =  file_list

    def file_move(self, move_path):
        ### 대상 파일 이동 ###
        for file in self.file_list:
            file_name = file.split('\\')[-1] # file_name = 파일명.type
            sh.move(file ,move_path + '/' + file_name)

        print('file move success.')

    def file_copy(self, copy_path):
        ### 대상 파일 복사 ###
        for file in self.file_list:
            file_name = file.split('\\')[-1] # file_name = 파일명.type
            sh.copy(file ,copy_path)

        print('file copy success.')

if __name__ == '__main__':
    path = '/home/djjin/Test/merge' # 폴더가 있어야 함.
    mv_path = '/home/djjin/Mywork/img2label/Yolo_mark/x64/Release/data/img'

    f = file_processing(path=path ,move_path = mv_path)
    f.find_file_path_lv1()
    f.find_file_path()
    #f.file_move(mv_path)
    f.file_copy(mv_path)
