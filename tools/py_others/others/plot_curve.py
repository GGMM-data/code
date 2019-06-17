import matplotlib.pyplot as plt

x = [
[1, 2, 4, 8],
[1, 2, 4, 8],
[1, 2, 4, 8],
[1, 2, 4, 8],
[1, 2, 4, 8],
[1, 2, 4, 8],
]
x_flatten = [
1, 2, 4, 8,
1, 2, 4, 8,
1, 2, 4, 8,
1, 2, 4, 8,
1, 2, 4, 8,
1, 2, 4, 8,
]

instruction=[
        [0.4537,0.4326,0.3783,0.3127],
        [0.4281,0.3843,0.3092,0.2138],
        [0.3881,0.3133,0.2449,0.1746],
        [0.3261,0.2458,0.1747,0.1081],
        [0.2681,0.1847,0.1039,0.0189],
        [0.1961,0.1031,0.0356,0.0145],
    ]
instruction_flatten=[
        0.4537,0.4326,0.3783,0.3127,
        0.4281,0.3843,0.3092,0.2138,
        0.3881,0.3133,0.2449,0.1746,
        0.3261,0.2458,0.1747,0.1081,
        0.2681,0.1847,0.1039,0.0189,
        0.1961,0.1031,0.0356,0.0145,
    ]
data=[
        [0.1900,0.0701,0.0380,0.0079],
        [0.1280,0.0441,0.0112,0.0051],
        [0.0592,0.0173,0.0053,0.0050],
        [0.0310,0.0097,0.0050,0.0050],
        [0.0156,0.0063,0.0050,0.0050],
        [0.0081,0.0050,0.0050,0.0050],
]
data_flatten=[
        0.1900,0.0701,0.0380,0.0079,
        0.1280,0.0441,0.0112,0.0051,
        0.0592,0.0173,0.0053,0.0050,
        0.0310,0.0097,0.0050,0.0050,
        0.0156,0.0063,0.0050,0.0050,
        0.0081,0.0050,0.0050,0.0050,
]

cache_size_index = [
    [4*2+0, 4*1+1, 4*0+2],
    [4*3+0, 4*2+1, 4*1+2, 4*0+3],
    [4*4+0, 4*3+1, 4*2+2, 4*1+3],
    [4*5+0, 4*4+1, 4*3+2, 4*2+3],
]

plot_labels = ["16 sets", "32 sets", "64 sets", "128 sets", "256 sets", "512 sets"]
scatter_labels = ['size = 1KB', 'size = 2KB', 'size = 4KB', 'size = 8KB']
scatter_markers = ['*', 's', '^', 'o']
colors=['y', 'g', 'r', 'b']


plt.figure()
for i in range(len(plot_labels)):
    plt.plot(x[i], data[i], label=plot_labels[i])

for i in range(len(cache_size_index)):
    x_list = []
    y_list = []
    for index in cache_size_index[i]:
        x_list.append(x_flatten[index])
        y_list.append(data_flatten[index])
    plt.scatter(x_list, y_list, color=colors[i], marker=scatter_markers[i], label=scatter_labels[i])

plt.legend()
plt.xlabel("Associativity")
plt.ylabel("Miss Ratio")
plt.title("Data Cache")
plt.savefig("./results/part1_data_miss_rate.png")

#=======================
plt.figure()
for i in range(len(plot_labels)):
    plt.plot(x[i], instruction[i], label=plot_labels[i])

for i in range(len(cache_size_index)):
    x_list = []
    y_list = []
    for index in cache_size_index[i]:
        x_list.append(x_flatten[index])
        y_list.append(instruction_flatten[index])
    plt.scatter(x_list, y_list, color=colors[i], marker=scatter_markers[i], label=scatter_labels[i])

plt.legend()
plt.xlabel("Associativity")
plt.ylabel("Miss Ratio")
plt.title("Instruction Cache")
plt.savefig("./results/part1_instruction_miss_rate.png")

plt.show()
