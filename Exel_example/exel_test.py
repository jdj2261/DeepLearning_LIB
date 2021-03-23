from math import radians
import openpyxl 
import matplotlib.pylab as plt
import pandas as pd
import random
import numpy as np

raw_data = {}

# # 현재 폴더 위치한 엑셀 파일 읽기
# wb = openpyxl.load_workbook('test.xlsx')

# # 엑셀 내 모든 시트 이름 출력
# print(wb.sheetnames)

# sheet = wb.active

# print(sheet.title)

# df = pd.read_excel('test.xlsx', engine='openpyxl')

# df.index = df['Time']
# # df['Velocity(kph)'].plot()
# df['ODO'].plot()
# # df['Voltage'].plot()
# # plt.legend(['Velocity(kph)', 'ODO', 'Voltage'])
# plt.grid(True)
# plt.show()

x = np.arange(0,1, 0.01)
# # print(x)
# y = [random.randrange(4,8,2) for _ in range(len(x))]
# # y = np.random.rand(len(x))
# plt.plot(x, y)
# plt.show()