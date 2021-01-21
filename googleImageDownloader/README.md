# googleImageDownloader

## Adrian Rosebrock의 구글 이미지 다운로더(javascript, python 사용)  
1. 이 깃을 다운로드 받는다.  
2. 크롬을 연다.  
3. 구글 검색에서 원하는 이미지를 검색한다.  
4. 이미지가 많이 로드 될 수 있도록 마우스를 아래로 스크롤 한다.  
5. Shift-Ctrl+i 버튼을 누른다음 옆에 열리는 창 윗부분에서 console을 클릭한다.  
6. 다운로드 한 js_console 파일을 메모장으로 열고 모든 내용을 선택하여 복사한다.  
7. 콘솔 창의 '>' 오른쪽을 클릭하고 붙여넣는다.  
8. urls.txt 파일이 다운로드 되면 깃 폴더의 urls.txt 파일에 덮어 씌운다.  
9. 윈도우의 왼쪽 하단 검색창을 클릭하고 cmd를 입력하여 커맨드 창을 연다.  
10. 깃 폴더의 상단에 폴더 주소창을 클릭하여 복사한 다음 커맨드 창에서 cd 명령어로 해당 폴더로 이동한다. 아래 예시는 내 컴퓨터의 위치이니 그대로 따라하지 말고 각자 자기 컴퓨터의 다운로드 받아서 압축을 푼 깃 폴더로 이동해야 한다.  
```python 
$ cd C:\Users\Camp51.9 Welcomezone\Documents\ATOM\google-images-deep-learning 
```
11. 이미지를 다운로드 받을 폴더(여기서는 images/santa)를 만든다.  
12. 아래 명령으로 파이썬을 실행하여 이미지를 다운로드 받는다. 다만 아래 명령어의 마지막 부분에 있는 images/santa 부분은 해당 폴더를 미리 만들어 놓아야 한다.    
```python 
$ python download_images.py --urls urls.txt --output images/santa
```
  
  
## 구글 드라이브를 이용한 방법은 아래 링크와 이미지를 참고하세요.
- 사본을 만들어서 자신의 구글 드라이브에서 사용하세요.  
- [구글 드라이브로 구글 이미지를 크롤링 하는 방법](https://colab.research.google.com/drive/11stLUm_jowYF0X12JiHrqkvlut2ONWEi?usp=sharing)  

![](https://github.com/mtinet/googleImageDownloader/blob/master/images/image%20downloader.png?raw=true)  
![](https://github.com/mtinet/googleImageDownloader/blob/master/images/copy%20path.png?raw=true)  
