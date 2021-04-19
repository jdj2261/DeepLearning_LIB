"""
Title : decorator Python Code
Date : 21. 04. 16
Author: Jin Dae Jong
"""

"""
1. 데코레이터는 함수를 파라미터로 전달 받으며, 그 함수 앞 뒤에 다른 동작들을 수행해준다.
2. 코드의 재사용성, 가독성을 높이기 위해 사용한다.
3. 프레임워크를 사용할 때 많이 볼 수 있다.
"""

def hello_decorator(func):
    """
    func을 파라미터로 전달 받는다.
    decorated라는 함수를 내부에 만들고 리턴
    """
    def decorated():
        print('Hello!')
        func()
        print('Nice to meet you!')
    return decorated

# 데코레이터는 장식시킬 함수의 앞과 뒤에서 코드를 실행
# 부분적으로 중복되는 작업을 수행할 경우, 코드의 재사용성을 높이기 위해 사용
@hello_decorator
def hello():
    print("I'm Jin")

hello()

from datetime import date, datetime

def timecost(func):
    def decorated():
        start = datetime.now()
        func()
        end = datetime.now()
        print('timecost', end - start)
    return decorated

import time

@timecost
def task1():
    time.sleep(3)
    print('this is task1')

@timecost
def task2():
    time.sleep(5)
    print('this is task2')

@timecost
def task3():
    time.sleep(2)
    print('this is task3')

# task1()
# task2()
# task3()

# 함수가 아닌 클래스로

class Timecost():
    def __init__(self, func):
        self.func = func
    
    def __call__(self):
        start = datetime.now()
        self.func()
        end = datetime.now()
        print('timecost', end-start)

@Timecost
def task1():
    time.sleep(3)
    print('this is task1')

@Timecost
def task2():
    time.sleep(5)
    print('this is task2')

@Timecost
def task3():
    time.sleep(2)
    print('this is task3')
    
task1()
task2()
task3()