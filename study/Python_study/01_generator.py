"""
Title : generator Python Code
Date : 21. 04. 16
Author: Jin Dae Jong
"""

# 특정 길이의 숫자 배열이 주어졌을 때, 이를 제곱한 수들을 담은 배열 출력
# 1. list 활용
num_count = 10
nums = [i for i in range(num_count)]
print(nums)

def sqaure_list(nums):
    result = []
    for num in nums:
        result.append(num * num)
    return result

result = sqaure_list(nums)
print(result)

# 2. generator 이용
def sqaure_generator(nums):
    for num in nums:
        yield num * num

result = sqaure_generator(nums)
print(result)
# <generator object sqaure_generator at 0x7ff5d97f4af0>

"""
return과 yield 차이
 - return : 임의의 수 반환
 - yield : 그 시점에서 일시 정지, 변수 기억 
            -> 원하는 값 리턴하며, 실행 흐름을 일시 정지하여 함수 재활용 가능
"""

import random
def randnum_return():
    a = random.randint(0, 100)
    for i in range(10):
        return a
        print('hello')

return_result = randnum_return()
print(return_result)
return_result = randnum_return()
print(return_result)
return_result = randnum_return()
print(return_result)

def randnum_yield():
    a = random.randint(0, 100)
    for i in range(10):
        yield a
        print('hello')

yield_result = randnum_yield()
print(yield_result)
print(next(yield_result))
print(next(yield_result))
print(next(yield_result))

def square_generator(nums):
    for num in nums:
        yield num * num

nums = [i for i in range(0, 10)]
result = sqaure_generator(nums)
for num in result:
    print(num)

result = (i * i for i in range(10))
print(result)
for num in result:
    print(num)