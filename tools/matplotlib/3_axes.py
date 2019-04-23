import matplotlib.pyplot as plt
import numpy as np


x = np.arange(-3.5,3.5,0.5)
y1 = np.abs(2 * x)
y2 = np.abs(x)

plt.figure(figsize=(10,10))
#plt.bar(x,y1,width=0.2)
plt.plot(x,y1,lw=2,marker='*',ms=8)
plt.plot(x,y2,lw=3,marker='^',ms=10)


# xlim and ylime
plt.xlim([-3.8, 3.3])
plt.ylim([0, 7.2])

# xlabel and ylabel
plt.xlabel('x',fontsize=20)
plt.ylabel('y = 2x ')

# xticklabel and yticaklabel
plt.xticks(x,('a','b','c','d','e','f','g','h','i','j','k','l','m','n'),fontsize=20)

plt.legend(['y1','y2'])
ax = plt.gca() # gca = get current axis
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('red')
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))
ax.legend(['t1','t2'])

plt.show()

