from matplotlib.table import Table
import matplotlib.pyplot as plt

plt.figure()
ax = plt.gca()
ax.set_axis_off()

table_values = [[1,2,3],[4,5,6],[5,6,1]]
row_labels = [1,2,3]
column_labels = [1,2,3]
column_widths = [0.3,0.3,0.3]

plt.table(cellText=table_values,                                                      colWidths=column_widths,cellLoc='center',                                   rowLabels=row_labels,rowLoc='center',                                       colLabels=column_labels,colLoc='center',                                    loc='center')

plt.show()

######################################################
plt.figure()
ax = plt.gca()
ax.set_axis_off()

tb = Table(ax,bbox=[0,0,1,1])

table_row = 3
table_col = 3
shapes = [table_row, table_col]
width,height = 1.0/table_col,1.0/table_row

for i in range(table_row):
  for j in range(table_col):
    loc = ('left' if i == 0 else 'center' if i == 1 else 'right')
    tb.add_cell(i,j,width,height,text=i+j,loc=loc)
for i in range(table_row):
    tb.add_cell(i,-1,width,height,text=i+1)
ax.add_table(tb)
plt.show()
