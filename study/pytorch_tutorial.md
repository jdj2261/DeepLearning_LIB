# Pytorch로 딥러닝하기

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)[![Git HUB Badge](http://img.shields.io/badge/-Tech%20blog-black?style=flat-square&logo=github&link=https://github.com/jdj2261)](https://github.com/jdj2261)

파이썬 딥러닝 파이토치 책 (pytorch 1.5.0, torchvision 0.6.0)

pytorch-tutorial 진행

> 참고
>
> [Pytorch 공식 사이트](https://tutorials.pytorch.kr/beginner/deep_learning_60min_blitz.html)
>
> [Pytorch 코드 작성 팁](https://sensibilityit.tistory.com/511)(한글 번역)

작성자 : 진대종 ([github](https://github.com/jdj2261))

> Environment
>
> - Ubuntu Version : 18.04
> - CUDA Version : 10.0
> - cuDNN Version :  7.6.5

## 1. MNIST로 MLP(Multi Layer Perceptron) 구현

### 1-1. MLP 모델 설계 순서

1. 모듈 임포트하기
2. 딥러닝 모델을 설계할 때 활용하는 장비 확인하기
3. MNIST 데이터 다운로드하기 (Train set, Test set 분리하기)
4. 데이터 확인하기
5. MLP 모델 설계하기
6. Optimizer, Objective Function 설정하기
7. 학습 데이터에 대한 모델 성능 확인하는 함수 정의하기
8. 검증 데이터에 대한 모델의 성능 확인하는 함수 정의하기
9. Train, Test set의 Loss 및 Test set Accuracy 확인하기

- Code

  ~~~python
  # 1. Module Import
  import numpy as np
  import matplotlib.pyplot as plt
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  from torchvision import transforms, datasets
  ~~~

  ---

  ~~~python
  # 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인
  if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
  else:
    DEVICE = torch.device('cpu')
  print('Using PyTorch version:', torch.__version__, 'Device:', DEVICE)
  >> Using PyTorch version: 1.5.0 Device: cuda
  ~~~

  ---

  ~~~python
  BATCH_SIZE = 32
  EPOCHS = 10
  ~~~

  - 전체 데이터가 10000개

    BATCH_SIZE = 1000 이면 

    1에포크 당 10회의 Iteration(학습 횟수) 

  ---

  ~~~python
  ''' 3. MNIST 데이터 다운로드 (Train set, Test set 분리하기) '''
  train_dataset = datasets.MNIST(root = "../data/MNIST",
                                 train = True,
                                 download = True,
                                 transform = transforms.ToTensor())
  
  test_dataset = datasets.MNIST(root = "../data/MNIST",
                                train = False,
                                transform = transforms.ToTensor())
  
  train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                             batch_size = BATCH_SIZE,
                                             shuffle = True)
  
  test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = False)
  ~~~

  - root : 데이터가 저장될 장소

  - train : 학습용인지(train = True), 검증용인지(train = False)

  - download : 다운로드해서 이용할 것인지

  - transform : 이미지 데이터 전처리 수행 

    ToTensor() 메서드를 이용해 0~1 범위로 정규화 과정 진행

  - DataLoader 함수를 이용해 Mini-Batch 별로 데이터를 묶음

  ---

  ~~~python
  ''' 4. 데이터 확인하기 (1) '''
  for (X_train, y_train) in train_loader:
      print('X_train:', X_train.size(), 'type:', X_train.type())
      print('y_train:', y_train.size(), 'type:', y_train.type())
      break
  # X_train: torch.Size([32, 1, 28, 28]) type: torch.FloatTensor
  # y_train: torch.Size([32]) type: torch.LongTensor
  ~~~

  - X_train : 32개의 데이터가 1개의 Mini-Batch를 구성하고 있고, 가로 28개, 세로 28개의 픽셀로 구성돼 있으며 채널이 1인 그레이스케일로 이뤄진 데이터
  - Y_train : 32개의 이미지 데이터 각각에 label값이 1개씩 존재하므로 32개의 값

  ---

  ~~~python
  ''' 5. 데이터 확인하기 (2) '''
  pltsize = 1
  plt.figure(figsize=(10 * pltsize, pltsize))
  for i in range(10):
      plt.subplot(1, 10, i + 1)
      plt.axis('off')
      plt.imshow(X_train[i, :, :, :].numpy().reshape(28, 28), cmap = "gray_r")
      plt.title('Class: ' + str(y_train[i].item()))
  ~~~

  - 그림으로 확인

  ---

  ~~~python
  ''' 6. Multi Layer Perceptron (MLP) 모델 설계하기 '''
  class Net(nn.Module): # nn.Module 클래스 상속
      def __init__(self):
          super(Net, self).__init__() 
          self.fc1 = nn.Linear(28 * 28, 512) # output의 노드 수 : 512
          self.fc2 = nn.Linear(512, 256)	   # Input:512, Output:256
          self.fc3 = nn.Linear(256, 10)      # Input:256, output:10
  
      def forward(self, x):			# Forward Propagation 
          x = x.view(-1, 28 * 28)		# 2차원 데이터를 1차원 데이터로 변환 (Flatten)
          x = self.fc1(x)				# 1차원으로 펼친 이미지 데이터 통과
          x = F.sigmoid(x)			# 두번째 Fully Connected Layer의 Input으로 계산
          x = self.fc2(x)				
          x = F.sigmoid(x)
          x = self.fc3(x)
          x = F.log_softmax(x, dim = 1) # 0부터 9까지, 총 10가지 경우의 수 중 하나로 분류하는 일 수행 
          return x
  ~~~

  - log_softmax() : Loss값에 대한 Gradient 값을 좀 더 원할하게 계산

  ---

  ~~~python
  ''' 7. Optimizer, Objective Function 설정하기 '''
  model = Net().to(DEVICE)	# CPU로 돌릴건지, GPU로 돌릴건지
  optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum = 0.5) # SGD Optimizer 정의, Learning Rate : 파라미터업데이트 시 반영, momentum : Optimizer 관성
  criterion = nn.CrossEntropyLoss() # 원-핫 인코딩 값
  print(model)
  
  """
  Net(
    (fc1): Linear(in_features=784, out_features=512, bias=True)
    (fc2): Linear(in_features=512, out_features=256, bias=True)
    (fc3): Linear(in_features=256, out_features=10, bias=True)
  )
  """
  ~~~

  ---

  ~~~python
  ''' 8. MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
  def train(model, train_loader, optimizer, log_interval):
      model.train() # 정의한 MLP 모델 학습 상태로 지정
      for batch_idx, (image, label) in enumerate(train_loader):
          image = image.to(DEVICE)
          label = label.to(DEVICE)
          optimizer.zero_grad() # optimizer Gradient 초기화
          output = model(image)
          loss = criterion(output, label) # CrossEntropy 이용하여 Loss 값 계산
          loss.backward() # Back Propagation으로 계산된 Gradient 각 파라미터에 할당
          optimizer.step() # 파리미터 값 업데이트
  
          if batch_idx % log_interval == 0:
              print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                  epoch, batch_idx * len(image), 
                  len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                  loss.item()))
  ~~~

  ---

  ~~~python
  ''' 9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
  def evaluate(model, test_loader):
      model.eval()		# 학습 상태가 아닌 평가 상태로 지정
      test_loss = 0		# Loss 값 계산을 위해 test_loss 임시 설정
      correct = 0			# 올바른 Class로 분류한 경우를 세기 위함
  
      with torch.no_grad():				# 평가하는 단계에서는 Gradient 업데이트 못하게 해야 함 -> Gradient 흐름 억제
          for image, label in test_loader: 
              image = image.to(DEVICE)
              label = label.to(DEVICE)
              output = model(image)
              test_loss += criterion(output, label).item()  	# CrossEntropy를 이용해 Loss 값을 계산한 결괏값 업데이트
              prediction = output.max(1, keepdim = True)[1]	# 계산된 벡터 값 내 가장 큰 값인 위치에 대응하는 클래스로 예측했다고 판단
              correct += prediction.eq(label.view_as(prediction)).sum().item() # 예측한 클래스 값과 실제값이 같으면 correct 횟수 저장
      
      test_loss /= (len(test_loader.dataset) / BATCH_SIZE) # 평균 loss 값 계산
      test_accuracy = 100. * correct / len(test_loader.dataset) # 얼마나 맞췄는지 계산해 정확도 계싼
      return test_loss, test_accuracy
  ~~~

  ---

  ~~~python
  ''' 10. MLP 학습 실행하며 Train, Test set의 Loss 및 Test set Accuracy 확인하기 '''
  for epoch in range(1, EPOCHS + 1):									# 학습 10번 진행
      train(model, train_loader, optimizer, log_interval = 200)		# MLP 모델, 학습 데이터, SGD
      test_loss, test_accuracy = evaluate(model, test_loader)			# 각 Epoch별로 출력되는 Loss값과 accuracy 값 계산
      print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
          epoch, test_loss, test_accuracy)) 							# 모니터링
  ~~~

## 2. 딥러닝의 발전을 이끈 알고리즘

### 2-1. Dropout

Weight Matrix에 랜덤하게 일부 Column에 0을 집어넣어 연산

**과적합을 방지하자 **

~~~python
''' 6. Multi Layer Perceptron (MLP) 모델 설계하기 '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5 # 몇 퍼센트의 노드를 계산 안할건지

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob) # model.train()로 명시할 때 self.trainig = True, model.eval()로 명시할 때 self.trainig = False로 할 것
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x
~~~

---

### 2-2. Activation 함수

#### 2-2-1. ReLU 함수

f(x) = max(0, x)

Activation 미분 값이 0또는 1이 되므로 아예 없애거나 완전히 살리는 것으로 해석

따라서 Layer가 깊어져도 Gradient Vanishing이 일어나는 것을 완화

~~~python
''' 6. Multi Layer Perceptron (MLP) 모델 설계하기 '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.relu(x) # sigmoind를 relu로 변경
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x
~~~

---

### 2-3. Batch Normalization

신경망에는 과적합과 Gradient Vanishing 외에도 **Internal Covariance shift** 현상 발생

각 Layer마다 Input 분포가 달라져 학습 속도가 느려지는 현상

**Batch Normalization**은 Layer의 Input 분포를 정규화해 학습 속도를 빠르게 하는것

Batch Normalization의 분포를 정규화해 비선형 활성 함수의 의미를 살리는 개념

~~~python
''' 6. Multi Layer Perceptron (MLP) 모델 설계하기 '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5
        self.batch_norm1 = nn.BatchNorm1d(512) # 첫번째 Fully Connected Layer의 Output이 512이므로
        self.batch_norm2 = nn.BatchNorm1d(256) # 두번째 Fully Connected Layer의 Output이 256이므로

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.batch_norm1(x) # 첫번째 Layer의 Output을 input으로 사용
        x = F.relu(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc2(x)
        x = self.batch_norm2(x) # 두번째 Layer의 Output을 input으로 사용
        x = F.relu(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x
~~~

---

### 2-4. Initialization

- 신경망을 어떻게 초기화하느냐에 따라 학습 속도가 달라진다.

~~~python
''' 7. Optimizer, Objective Function 설정하기 '''
import torch.nn.init as init # Weight, Bias등 딥러닝 모델에서 초깃값으로 설정되는 요소에 대한 모듈
def weight_init(m):	# Weight 초기화
    if isinstance(m, nn.Linear):	# nn.Linear에 해당하는 파라미터 값에 대해서만 
        init.kaiming_uniform_(m.weight.data) # he_initialization을 이용해 파라미터 값 초기화

model = Net().to(DEVICE)
model.apply(weight_init) # Net() 클래스 인스턴스인 모델에 적용
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.CrossEntropyLoss()

print(model)
~~~

- LeCun Initialization 
- He Initialization 

---

### 2-5. Optimizer

- SGD

  Batch 단위로 Back Propagation하는 과정 중 하나

- Momentum

  미분을 통한 Gradient 방향으로 가되, 일종의 관성을 추가

  > SGD는 최적해를 찾아가며, 전체 데이터에 대해 Back Propagation을 하는 것이 아니라 Batch 단위로 Back Propagation하기 때문에 일직선으로 찾아가지 않는다.
  >
  > Momentum은 걸어가는 보폭을 크게 하는 개념이라 이해하면 되고, Local Minimum을 지나칠 수 있는 장점이 있다.

- Adagrad

  가보지 않은 곳은 많이 움직이고, 가본 곳은 조금씩 움직이자.

  단점은 학습이 오래 진행될수록 부분이 계속 증가해 STep Size가 작아진다.

- RMSProp

  Adagrad 단점 보완 

- Adam(Adaptive Moment Estimation)

  모델을 디자인할 때 가장 많이 사용하는 Optimizer

  RMSProp와 Momentum 방식의 특징을 결합

~~~python
''' 7. Optimizer, Objective Function 설정하기 '''
import torch.nn.init as init
def weight_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight.data)

model = Net().to(DEVICE)
model.apply(weight_init)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)  #Adam 적용
criterion = nn.CrossEntropyLoss()

print(model)
~~~

## 3. CNN

### 3-1. CIFAR-10 데이터를 이용해 MLP 설계하기

~~~python
''' 1. Module Import '''
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

''' 2. 딥러닝 모델을 설계할 때 활용하는 장비 확인 '''
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

BATCH_SIZE = 32
EPOCHS = 10
~~~

~~~python
''' 3. CIFAR10 데이터 다운로드 (Train set, Test set 분리하기) '''
train_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                  train = True,
                                  download = True,
                                  transform = transforms.ToTensor())

test_dataset = datasets.CIFAR10(root = "../data/CIFAR_10",
                                train = False,
                                transform = transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = BATCH_SIZE,
                                            shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False)
~~~

~~~python
''' 6. Multi Layer Perceptron (MLP) 모델 설계하기 '''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x 
~~~

~~~python
''' 7. Optimizer, Objective Function 설정하기 '''
model = Net().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

print(model)
~~~

~~~python
''' 8. MLP 모델 학습을 진행하며 학습 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                epoch, batch_idx * len(image), 
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), 
                loss.item()))
~~~

~~~python
''' 9. 학습되는 과정 속에서 검증 데이터에 대한 모델 성능을 확인하는 함수 정의 '''
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    
    test_loss /= (len(test_loader.dataset) / BATCH_SIZE)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy
~~~

~~~python
''' 10. MLP 학습 실행하며 Train, Test set의 Loss 및 Test set Accuracy 확인하기 '''
for epoch in range(1, EPOCHS + 1):
    train(model, train_loader, optimizer, log_interval = 200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(
        epoch, test_loss, test_accuracy))
# [EPOCH: 10], 	Test Loss: 1.4587, 	Test Accuracy: 48.21 % 
~~~



