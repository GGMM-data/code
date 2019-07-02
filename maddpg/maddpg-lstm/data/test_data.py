import h5py as h5

f1 = h5.File("chengdu_1.h5", "r")
f2 = h5.File("chengdu_2.h5", "r")

data1 = f1['data'][:]
data2 = f2['data'][:]

f1.close()
f2.close()
print(data1.max())
print(data2.max())
