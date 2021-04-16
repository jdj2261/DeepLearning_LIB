# Python Interview

[[100 Essential Python Interview]](https://www.techbeamers.com/python-interview-questions-programmers/) 를 중심으로 정리할 예정이며 중요하다고 생각되는 것 위주로 정리하고자 합니다.

한재엽님의 [Interview_Question_for_Beginner/Python](https://github.com/JaeYeopHan/Interview_Question_for_Beginner/tree/master/Python) 과 [python_study](https://github.com/yeomko22/python_study/blob/master/week2_decorator.ipynb) 를 참고하여 파이썬 코드를 작성하려 합니다.



## 1. 100가지의 파이썬 필수 질문

<details>
  <summary>Q-1. Python이란 무엇이며, 어떤 이점이 있고 PEP 8에 대해 무엇을 이해하고 있나?</summary>

- Python은 가장 성공적인 Interpreted 언어이며, 실행하기 전에 컴파일이 필요 없음.

- Python 프로그래밍의 이점

  - 변수의 데이터 타입을 언급 할 필요가 없음.

  - 객체 지향 프로그래밍을 지원.

  - 파이썬의 함수는 클래스 객체와 같음. 변수에 할당하고 다른 메서드에서 반환하고 인수로 전달할 수 있음

  - 개발시간은 빠르지만 컴파일된 언어보다 속도는 느릴수가 있음. 

    다행히, C언어로 확장할 수 있어 최적화할 수 있음.

  - 웹 기반 애플리케이션, 테스트 자동화, 데이터 모델링, 빅데이터 분석 등 여러 용도로 사용함.

- PEP8은 파이썬 코딩 표준이며, 더 읽기 쉬운 코드를 제공하도록 안내함.

</details>

<details>
  <summary>Q-6. 버그를 찾거나 정적 분석을 할 수 있는 어플리케이션이 있는가?
    </summary>

- PyChecker : 정적 분석에 사용, 프로젝트의 버그를 식별하고 버그와 관련된 스타일 및 복잡성도 보여줌
- Pylint : 코딩 표준을 충족하는지 확인

</details>

<details>
  <summary>Q-7. decorator는 언제 사용되는가?
    </summary>

- 함수를 빠르게 변경할 때 사용 가능함.

</details>

<details>
  <summary>Q-8. 리스트와 튜플의 차이점은?
    </summary>

- 리스트는 mutable(수정 가능), 튜플은 immutable(수정 불가능)

</details>

<details>
  <summary>Q-9. 파이썬에서 메모리는 어떻게 관리되는가?
    </summary>

- 개별 힙을 사용하여 메모리 유지
- 힙은 모든 파이썬 객체와 자료구조를 가지고 있으며 이 영역은 파이썬 인터프리터만이 접근 가능하며 프로그래머는 사용 불가능
- 내장된 가비지 컬렉터를 통해 사용되지 않은 메모리 관리

</details>

<details>
  <summary>Q-10. lambda와 def의 주요 차이점은 무엇인가?
    </summary>

- def는 여러 표현식을 가질 수 있지만, lambda는 단일 함수
- def는 함수를 생성하고 나중에 호출 할 이름을 지정하고, lambda는 함수 객체를 형성하고 반환
- def는 return문을 가질 수 있지만 lambda는 불가능
- lambda는 list나 dictionary에서 사용가능

</details>

<details>
  <summary>Q-21. Docstring은 무엇인가?
    </summary>

- 파이썬의 모듈, 함수, 클래스, 메소드 정의의 첫 번째 명령문으로 발생하는 문자열 리터럴
- 해당 객체의 doc 특수 속성으로 변환됨

~~~python
def print_items(items):
	# Doctsring (print_items.__doc__)
    """
    items를 print
    :param items: 
    :return: 
    """
    for item in items:
        print(item)
~~~

</details>

<details>
  <summary>Q-26. return 키워드는 무엇인가?
    </summary>

- 함수의 목적은 입력을 받아 출력을 반환하는 것
- return은 호출자에게 값을 보내는데 사용

</details>

<details>
  <summary>Q-27. Call by Value란?
    </summary>

- 표현식 또는 값이 함수의 각 변수에 바인딩되는지 여부를 나타내는 인수
- 해당 변수는 로컬로 취급하며, 함수 외부에 반영되지 않음.

</details>

<details>
  <summary>Q-28. call by reference란?
    </summary>

- 참조로 인수를 전달하면 단순 복사가 아닌 함수에 대한 암시적 참조로 사용됨.
- 로컬 복사본을 만들 필요가 없으므로 시간과 공간 효율성을 높일 수 있음.
- 함수 호출 중 변수가 실수로 변경 될 수 있으므로 프로그래머는 이러한 불확실성을 피하기 위한 코드를 작성해야 함.

</details>

<details>
  <summary>Q-33. *Args는 무엇을 하는가?
    </summary>

- N개의 매개변수를 넘기겠다.

  ~~~python
  # Python code to demonstrate 
  # *args for dynamic arguments 
  def fn(*argList):  
      for argx in argList:  
          print (argx) 
      
  fn('I', 'am', 'Learning', 'Python')
  ~~~

  ~~~python
  # Output
  I
  am
  Learning
  Python
  ~~~

</details>

<details>
  <summary>Q-34. **kwargs는 무엇을 하는가?
    </summary>

- 이름이나 키워드로 지정할 수 있는 N 개의 인수를 전달

  ~~~python
  # Python code to demonstrate 
  # **kwargs for dynamic + named arguments 
  def fn(**kwargs):  
      for emp, age in kwargs.items(): 
          print ("%s's age is %s." %(emp, age)) 
      
  fn(John=25, Kalley=22, Tom=32)
  ~~~

  ~~~python
  # output
  John's age is 25.
  Kalley's age is 22.
  Tom's age is 32.
  ~~~

</details>

<details>
  <summary>Q-39. pass와 continue의 차이는?
    </summary>

- pass문은 아무것도 하지 않는다.
- continue문은 루프가 다음 반복에서 다시 시작되도록 한다.

</details>

<details>
  <summary>Q-51. GIL이란?
    </summary>

- Global Interpreter Lock의 약자로 인터프리터가 한 스레드만 하나의 바이트코드를 실행 시킬 수 있도록 해주는 Lock

- 파이썬은 기본적으로 garbage collection과 reference counting을 통해 할당된 메모리를 관리하는데

  멀티스레드인 경우 여러 스레드가 하나의 객체를 사용한다면 reference count를 관리하기 위하여 모든 객체에 대한 lock이 필요함

  이러한 비효율을 막기 위해 GIL을 사용하게 됨

- 하나의 Lock을 통해 모든 객체들에 대한 reference count의 동기화 문제를 해결

</details>

<details>
  <summary>Q-53. 파이썬에서는 메모리를 어떻게 관리하는가?
    </summary>

- 모든 객체와 자료구조를 가지고 있는 힙 관리자가 내부적으로 구현되어 있음.
- 이 힙 관리자는 객체애 대한 힙 공간 할당, 할당 해제를 수행함.

</details>

<details>
  <summary>Q-54. 튜플이란?
    </summary>

- immutable한 자료형으로 리스트와 비슷한 구조를 가지고 있지만, 수정이 가능하냐 안하느냐는 차이가 있음

</details>

<details>
  <summary>Q-55. dictionary란?
    </summary>

- collection 데이터 타입의 하나로 key와 value의 구조로 이뤄진 데이터 타입
- 해쉬, 맵 혹은 해쉬맵이라고 불림.

</details>

<details>
  <summary>Q-56. Set?
    </summary>

- 순서가 없는 collection 데이터 객체로 유니크하고 변경 불가능한 객체를 저장

</details>

<details>
  <summary>Q-58. 파이썬의 리스트는 연결리스트 인가?
    </summary>

- 가변 길이의 배열로 C 스타일의 연결리스트와는 다르다.
- 내부적으로 다른 객체를 참조하기 위한 연속적인 배열을 가지며, 배열 변수에 대한 포인터와 그 길이를 리스트 헤더에 저장

</details>

<details>
  <summary>Q-63. 파이썬에서 Composition이란?
    </summary>

- 상속의 한 종류로 기본 클래스에서 상속을 하지만 파생 클래스의 멤버 역할을 하는 기본 클래스의 인스턴스 변수를 사용

- 내부적으로 다른 객체를 참조하기 위한 연속적인 배열을 가지며, 배열 변수에 대한 포인터와 그 길이를 리스트 헤더에 저장

  ~~~python
  class PC: # Base class
      processor = "Xeon" # Common attribute
      def __init__(self, processor, ram):
          self.processor = processor
          self.ram = ram
  
      def set_processor(self, new_processor):
          processor = new_processor
  
      def get_PC(self):
          return "%s cpu & %s ram" % (self.processor, self.ram)
  
  class Tablet():
      make = "Intel"
      def __init__(self, processor, ram, make):
          self.PC = PC(processor, ram) # Composition
          self.make = make
  
      def get_Tablet(self):
          return "Tablet with %s CPU & %s ram by %s" % (self.PC.processor, self.PC.ram, self.make)
  
  if __name__ == "__main__":
      tab = Tablet("i7", "16 GB", "Intel")
      print(tab.get_Tablet())
  ~~~

</details>

<details>
  <summary>Q-66. 미리 정의된 조건에 대한 예외를 어떻게 발생시키는가?
    </summary>

~~~python
# Example - Raise an exception
while True:
    try:
        value = int(input("Enter an odd number- "))
        if value%2 == 0:
            raise ValueError("Exited due to invalid input!!!")
        else:
            print("Value entered is : %s" % value)
    except ValueError as ex:
        print(ex)
        break
~~~

~~~python
# output
Enter an odd number- 2
Exited due to invalid input!!!
# output
Enter an odd number- 1
Value entered is : 1
Enter an odd number-
~~~

</details>

<details>
  <summary>Q-69. generator?
    </summary>

- iterator를 생성하는 함수이며, yield 키워드를 사용하여 반환
- return과 달리 반환 후에 종료되지 않고 그 상태 유지
- memory를 효율적으로 사용할 수 있다.

</details>

<details>
  <summary>Q-70. Closures?
    </summary>

- 함수 객체로서 다른 함수를 반환

  ~~~python
  def multiply_number(num):
      def product(number):
          'product() here is a closure'
          return num * number
      return product
  
  num_2 = multiply_number(2)
  print(num_2(11))
  print(num_2(24))
  
  num_6 = multiply_number(6)
  print(num_6(1))
  ~~~

  ~~~python
  # output
  22
  48
  6
  ~~~

</details>

<details>
  <summary>Q-71. Decorator?
    </summary>

- 함수 객체에 동적으로 새로운 기능을 추가

  ~~~python
  def decorator_sample(func):
      def decorator_hook(*args, **kwargs):
          print("Before the function call")
          result = func(*args, **kwargs)
          print("After the function call")
          return result
      return decorator_hook
  
  @decorator_sample
  def product(x, y):
      "Function to multiply two numbers."
      return x * y
  
  print(product(3, 3))
  ~~~

  ~~~python
  # output
  Before the function call
  After the function call
  9
  ~~~

</details>

<details>
  <summary>Q-84. zip() 메서드를 사용하는 방법?
    </summary>

- 여러 컨테이너의 해당 인덱스를 매핑하여 단일 단위로 사용함.

  ~~~python
  # Example: zip() function
    
  emp = [ "tom", "john", "jerry", "jake" ] 
  age = [ 32, 28, 33, 44 ] 
  dept = [ 'HR', 'Accounts', 'R&D', 'IT' ] 
    
  # call zip() to map values 
  out = zip(emp, age, dept)
    
  # convert all values for printing them as set 
  out = set(out) 
    
  # Displaying the final values  
  print ("The output of zip() is : ",end="") 
  print (out)
  ~~~

  ~~~python
  # output
  The output of zip() is : {('jerry', 33, 'R&D'), ('jake', 44, 'IT'), ('john', 28, 'Accounts'), ('tom', 32, 'HR')}
  ~~~

</details>

<details>
  <summary>Q-88. 객체를 복사하는 방법은?
    </summary>

- copy.copy()
  - 얕은 복사는 새로운 객체(변수)를 만든 후에 원본에 접근할 수 있는 참조(reference)를 입력한다.
    👉🏽 이런 경우 서로 다른 변수명이지만 본질적으로 서로 같은 대상을 의미하므로 하나의 변수 역시 수정이 된다.
  - 가변형(mutable) 자료형에 대해서 적용이 가능하다.
    👉🏽 가변형(mutable) 자료형은 같은 주소에서 값(value)이 변경 가능하기 때문에 얕은 복사가 가능하다.
    👉🏽 불변형(immutable) 자료형은 본질적으로 변경이 불가능하므로 재배정을 통해 변수를 바꾼다. 따라서 재배정이 이루어지므로 객체가 서로 달라진다.
- copy.deepcopy()
  - 깊은 복사는 내부에 객체들까지 모두 새롭게 copy 되는 것
  - 깊은 복사는 새로운 객체(변수)를 만든 뒤에 원본의 복사본을 변수에 입력한다.
    👉🏽 서로 값만 같을 뿐 본질적으로 서로 다르기 때문에 한 변수가 수정될 시 다은 변수가 수정되지 않는다.

</details>