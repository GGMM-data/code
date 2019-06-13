import h5py as h5

f1 = h5.File("TaxiDD_chengdu104.05_104.09_30.66_30.69_1800_32_32_2_count.h5", "r")
f2 = h5.File("TaxiDD_chengdu104.05_104.09_30.69_30.72_1800_32_32_2_count.h5", "r")

data1 = f1['data'][:]
data2 = f2['data'][:]

print(data1.max())
print(data2.max())
