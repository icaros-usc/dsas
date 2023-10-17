import matplotlib.pyplot as plt
import csv


y1 = []
y2 = []

with open('human_output.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        y1.append(float(row[0]))

with open('robot_output.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        y2.append(float(row[0]))

plt.plot(y1, label='human')
plt.plot(y2, label='robot')
#plt.xlabel('x')

plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
