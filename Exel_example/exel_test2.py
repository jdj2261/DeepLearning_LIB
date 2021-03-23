import random

Column1 = list(range(1,10))
Column2 = list(random.randrange(4,8,2) for _ in range(len(Column1)))


from xlsxwriter import Workbook
workbook = Workbook('Ecl.xlsx')
Report_Sheet=workbook.add_worksheet()
Report_Sheet.write(0, 0, 'Column1')
Report_Sheet.write(0, 1, 'Column2')

for row_ind, row_value in enumerate(zip(Column1, Column2)):
    print (row_ind, row_value)
    for col_ind, col_value in enumerate(row_value):
        Report_Sheet.write(row_ind + 1, col_ind, col_value)

workbook.close()