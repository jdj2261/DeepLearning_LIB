# Python Deep Learning Pytorch

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)[![Git HUB Badge](http://img.shields.io/badge/-Tech%20blog-black?style=flat-square&logo=github&link=https://github.com/jdj2261)](https://github.com/jdj2261)

파이썬 딥러닝 파이토치 책 정리 ([pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3))

작성자 : 진대종 ([github](https://github.com/jdj2261))

> Environment
>
> - Ubuntu Version : 18.04
> - CUDA Version : 10.1
> - cuDNN Version :  7.6.5



## 1. 환경설정

### 1-1. 가상환경 생성 및 활성화

~~~
$ conda create --name Python-DL-Pytorch
$ conda activate Python-DL-Pytorch
~~~

### 1-2. pytorch 설치하기

- pytorch 설치

  - [pytorch-1.4.0 설치](https://pytorch.org/get-started/previous-versions/)

  ~~~
  # CUDA 10.1
  $ conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
  ~~~

### 1-3. jupyter notebook 연동

참고 : [아나콘다 쥬피터 연동](https://yganalyst.github.io/pythonic/anaconda_env_2/)

- conda 환경에서 jupyter 연동을 위해  ipykernel을 설치해야 한다.

  ~~~
  $ conda install ipykernel
  ~~~

- 가상환경에 kernel을 연결하여 jupyter notebook과 연결시킨다.

  ~~~
  $ python -m ipykernel install --user --name Python-DL-Pytorch --display-name "pytorch-test"
  $ jupyter-notebook
  ~~~

- jupyter kernel 삭제

  ~~~
  $ jupyter kernelspec uninstall kernel명
  ~~~

## 2. 반드시 알아야 하는 파이토치 스킬

### 2-1. 텐서

- 텐서(Tensor)란, '데이터를 표현하는 단위'

#### 2-1.1. Scalar

- 흔히 알고 있는 상숫값

- 사칙연산을 이용해 계산할 수 있으며 torch 모듈에 내장된 메서드를 이용할 수 도 있다.

  ~~~python
  torch.add(scalar1, scalar2)
  torch.sub(scalar1, scalar2)
  torch.mul(scalar1, scalar2)
  torch.div(scalar1, scalar2)
  ~~~

#### 2-1.2. Vector

- 하나의 값을 표현할 때 2개 이상의 수치로 표현한 것

  스칼라의 형태와 동일한 속성을 갖고 있지만, 여러 수치 값을 이용해 표현한다.

- 스칼라 사칙연산과 동일한 방식이며, 벡터 간 내적 연산이 가능하다.

  ~~~
  torch.dot(vector1, vector2) # 벡터 내적 계산
  ~~~

#### 2-1.3. Matrix 

- 2개 이상의 벡터 값을 통합해 구성된 값, 선형 대수의 기본 단위

- 스칼라 벡터 사칙연산과 동일한 방식이며, 행렬의 곱 연산이 가능하다

  ~~~python
  # 행렬의 곱
  torch.matmul(matrix1, matrix2)
  ~~~

#### 2-1.4. Tensor

- 행렬을 2차원의 배열이라 표현할 수 있다면, 텐서는 2차원 이상의 배열이라 표현할 수 있다.

  ![img](https://miro.medium.com/max/1008/1*pUr-9ctuGamgjSwoW_KU-A.png)

### 2-2. Autograd

- 파이토치를 이용해 코드를 작성할 시 Back Propagation을 이용해 파라미터를 업데이트 하는 방법은 **Autograde** 방식으로 쉽게 구현 가능하다.

#### 2-2.1. torch import

~~~python
import torch
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
~~~

- GPU를 쓸건지 CPU를 쓸건지 선택

#### 2-2.2. 학습 파라미터

~~~
BATCH_SIZE = 64
INPUT_SIZE = 1000
HIDDEN_SIZE = 100
OUTPUT_SIZE = 10
~~~

- BATCH_SIZE : 파라미터를 업데이트할 때 계산되는 데이터 개수

  Input으로 이용되는 데이터가 64개라는 것을 의미

- INPUT_SIZE : Input의 크기이자 입력층의 노드 수

  BATCH_SIZE가 64이므로 1000 크기의 벡터 값을 64개 이용

  모양으로 설명하면 (64, 1000)이 됨

- HIDDEN_SIZE : Input을 다수의 파라미터를 이용해 계산한 결과에 한 번 더 계산되는 파라미터 수

  입력층에서 은닉층으로 전달됐을 때 은닉층의 노드 수 

  이 예제에서는 (64, 1000)의 Input들이 (1000, 100) 크기의 행렬과 행렬 곱을 계산하기 위해 설정 됨.

- OUTPUT_SIZE : 최종으로 출력되는 값의 벡터의 크기

  보통 최종으로 비교하고자 하는 레이블의 크기와 동일하게 설정한다.

  예를들어 10개로 분류하려면 10짜리의 원-핫 인코딩을 이용하기 때문에 Output의 크기를 '10'으로 맞추기도 하며 5 크기의 벡터 값에 대해 MSE를 계산하기 위해 Output의 크기를 '5'로 맞추기도 한다.

#### 2-2.3. Autograd 작동 방식

~~~python
x = torch.randn(BATCH_SIZE, INPUT_SIZE, device = DEVICE, dtype = torch.float, requires_grad = False) #(1)
y = torch.randn(BATCH_SIZE, OUTPUT_SIZE, device = DEVICE, dtype = torch.float, requires_grad = False) #(2)
w1 = torch.randn(INPUT_SIZE, HIDDEN_SIZE, device = DEVICE, dtype = torch.float, requires_grad = True) #(3)
w2 = torch.randn(HIDDEN_SIZE, OUTPUT_SIZE, device = DEVICE, dtype = torch.float, requires_grad = True) #(4)
~~~

(1) torch.randn : 평균이 0, 표준편차가 1인 정규분포에서 샘플링 한 값

x = (64, 1000) 모양의 데이터가 생성된다.

Input으로 이용되기 때문에 Gradient를 계산할 필요가 없다. 따라서

requires_grad = False로 설정한다.

(2) Output 역시 BATCH_SIZE 수만큼 결과 값이 필요하며 OUTPUT과의 오차를 계산하기 위해 Output의 크기를 '10'으로 설정한다.

(3) w1은 Input 데이터 크기가 1000이며 행렬 곱을 위해 다음 행 값이 1000 이어야 한다.

100 크기의 데이터를 생성하기 위해 (1000,100) 크기의 데이터를 생성

Gradient를 계산해야 하기 떄문에 requires_grad = True로 설정한다.

(4) Output을 계산할 수 있도록 (100,10) 행렬의 크기를 설정한다.

Back Propagation을 통해 업데이트해야 하는 대상이므로 requires_grad = True로 설정한다.

#### 2-2.4. 파라미터 업데이트

~~~python
learning_rate = 1e-6
for t in range(1, 501):
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 0:
        print("Iteration: ", t, "\t", "Loss: ", loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
~~~

- learning_rate를 어떻게 설정하느냐에 따라 Gradient 값에 따른 학습 정도가 결전된다.

  딥러닝 모델에서 파라미터 값을 업데이트할 때 **가장 중요한 하이퍼파라미터**이기도 하다.

- y_pred : 예측값 x와 w1간의 행렬 곱을 이용해 나온 결과를 계산 후 clamp라는 메서드를 이용해 비선형 함수를 적용 (ReLU)와 같은 역할

- loss : 예측값과 실제 레이블 값을 비교해 오차를 계산한 값

  제곱차의 합을 계산함.

- backward() : Back Propagation 진행

- with torch.no_grad() : Gradient를 계산한 결과를 이용해 파라미터 값을 업데이트 할 때, 해당 시점의 Gradient 값을 고정한 후 업데이트를 진행함.

  Gradient 값을 고정한다는 의미

- 음수 사용 이유 : Loss 값이 최소로 계산될 수 있는 파라미터 값을 찾기 위해 Gradient 값에 대한 반대 방향으로 계산한다는 것을 의미한다.

- 각 파라미터를 업데이트 했다면 Gradient를 초기화해 다음 반복문을 진행 할 수 있도록 Gradient 값을 0으로 설정함.

#### 2-2.5. 결과

~~~
Iteration:  100 	 Loss:  239.2540283203125
Iteration:  200 	 Loss:  0.456472247838974
Iteration:  300 	 Loss:  0.0020911425817757845
Iteration:  400 	 Loss:  9.306118590757251e-05
Iteration:  500 	 Loss:  2.3576532839797437e-05
~~~

---

