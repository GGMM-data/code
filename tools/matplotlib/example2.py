import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0,10,1)

y1 = x**2
y2 = 2*x +5
y3 = 4 * x
y4 = x**3

######
# 1.imshow

plt.plot(x,y1)
plt.show()

plt.plot(x,y2)
plt.show()

######
# 2.figure
plt.figure()
plt.plot(x,y1)

plt.figure()
plt.plot(x,y2)
plt.show()

######
# 3.plt.subplot
plt.figure()

plt.subplot(2,2,1)
plt.plot(x,y1)

plt.subplot(2,2,2)
plt.plot(x,y2)

plt.subplot(2,2,3)
plt.plot(x,y3)

plt.subplot(2,2,4)
plt.plot(x,y4)

plt.show()

#### 
# error! ax object has no attributes 
# ax subplot
#plt.figure()
#ax = plt.gca()
#ax.subplot(2,2,1)
#ax.plot(x,y1)
#
#ax.subplot(2,2,2)
#ax.plot(x,y2)
#
#ax.subplot(2,2,3)
#ax.plot(x,y3)
#
#ax.subplot(2,2,4)
#ax.plot(x,y4)
#
#plt.show()


######
# plt.subplots()
# plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw)
fig, ax = plt.subplots(2, 3, figsize=(40,20))

plt.show()
