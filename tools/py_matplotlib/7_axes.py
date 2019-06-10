import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-3.5,3.5,0.5)
y1 = np.abs(2 * x)
y2 = np.abs(x)

plt.figure(figsize=(10,10))
ax = plt.gca() # gca = get current axis
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('red')
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

# both work
ax.plot(x,y1,lw=2,marker='*',ms=8)
plt.plot(x,y2,lw=3,marker='^',ms=10)

# xlim and ylim
# ax.xlim([-3.8, 3.3])
# AttributeError: 'AxesSubplot' object has no attribute 'xlim'
plt.xlim([-3.8, 3.3])
plt.ylim([0, 7.2])

# xlabel and ylabel
# ax.xlabel('x',fontsize=20)
# AttributeError: 'AxesSubplot' object has no attribute 'xlabel'
plt.xlabel('x',fontsize=20)
plt.ylabel('y = 2x ')

# xticklabel and yticaklabel
# ax.xticks(x,('a','b','c','d','e','f','g','h','i','j','k','l','m','n'),fontsize=20)
# AttributeError: 'AxesSubplot' object has no attribute 'xticks'
plt.xticks(x,('a','b','c','d','e','f','g','h','i','j','k','l','m','n'),fontsize=20)

# both work
ax.legend(['t1','t2'])
plt.legend(['y1','y2'])

plt.show()

