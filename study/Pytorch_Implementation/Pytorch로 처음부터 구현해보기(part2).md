# Pytorch로 처음부터 구현해보기 (Part2)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)[![Git HUB Badge](http://img.shields.io/badge/-Tech%20blog-black?style=flat-square&logo=github&link=https://github.com/jdj2261)](https://github.com/jdj2261)

pytorch-yolo-v3 설치 및 데모 실행하기 

> 참고
>
> [How to implement a YOLO(v3) object detector from scratch in PyTorch ](https://github.com/ayooshkathuria/pytorch-yolo-v3)(영어)
>
> [YOLO(v3) object detector를 Pyotrch로 처음부터 구현해보기](https://devjin-blog.com/yolo-part1/)(한글 번역)

작성자 : 진대종 ([github](https://github.com/jdj2261))

> Environment
>
> - Ubuntu Version : 18.04
> - CUDA Version : 10.0
> - cuDNN Version :  7.6.5

## 1. Prerequisites

- YOLO가 어떻게 작동하는지 알야아 한다.

- Pytorch에 대한 기본 지식, nn.Module, nn.Sequential, torch.nn.parameter 클래스로 커스텀 구조를 어떻게 구현하는지 알아야 한다.

- Pytorch 경험이 없다면 공식 튜토리얼에서 Pytorch를 한번 다뤄보자.

  

## 3. Further Reading

1. [YOLO V1: You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
2. [YOLO V2: YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
3. [YOLO V3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
4. [Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
5. [Bounding Box Regression (Appendix C)](https://arxiv.org/pdf/1311.2524.pdf)
6. [IoU](https://www.youtube.com/watch?v=DNEm4fJ-rto)
7. [Non-maximum Suppression](https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH)
8. [PyTorch Official Tutorial](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)



## Terms

- Upsampling은 신호의 샘플링 주기를 높이는 것이다. [1 2 3 4 5]의 신호가 있을 때 샘플링 주기를 3배로 업샘플링한다면 [1 0 0 2 0 0 3 0 0 4 0 0 5 0 0]이 된다. 업샘플링 후에는 LPF(Low Pass Filter)로 인터폴레이션(Interpolation)을 하여야 한다.

- Downsampling은 신호의 샘플링 주기를 낮추는 것이다. [1 2 3 4 5 6 7 8 9 10 11 12]의 신호가 있을 때 샘플링 주기를 1/3로 다운샘플링한다면 [1 4 7 10]이다. 다운샘플링 전에는 LPF로 주파수를 낮춰야 한다. 다운샘플링을 데시메이션(Decimation)이라고도 한다.

- pooling 사용 이유 

  이미지의 크기를 계속 유지한 채 레이어로 가게 된다면 연산량이 기하급수적으로 늘게되며, 적당량의 data만 있어도 된다. 

  parameter를 줄이기 때문에, 해당 network의 표현력이 줄어들어 Overfitting을 억제하고 hardware resource를 절약하고 속도가 향상된다.

  최대 단점은 방향이나 비율이 달라지면 서로 다른 객체로 인식한다. 이를 개선하고자 다양한 방식이 나오게 되지만 학습시간이 증가하는 단점이 있다.

- stride는 필터가 이동할 간격을 말한다.

- [softmax](https://yamalab.tistory.com/87)