import os
import shutil
import pandas as pd
import re


class deal_didi:
    # def __init__(self):

    # 处理所有的原始问题
    def deal_all_file(self, pattern1, parent_dir1, pattern2=None, parent_dir2=None, parent_dir=None):
        # if os.path.exists(parent_dir + "single_driver_data"):
        #     shutil.rmtree(parent_dir + "single_driver_data")
        # os.mkdir(parent_dir + "single_driver_data")

        files = self.find_file_path(pattern, parent_dir1)
        print(files)
        for file_path in files:
            file_path = parent_dir1 + file_path
            print(file_path)
            self.deal_one_file(parent_dir, file_path)

        if parent_dir2 != None:
            files = self.find_file_path(pattern2, parent_dir2)
            print(files)
            for file_path in files:
                file_path = parent_dir2 + file_path
                print(file_path)
                self.deal_one_file(parent_dir, file_path)

    # 处理单个文件的内部逻辑
    def deal_one_file(self, parent_dir, file_path):
        data = pd.read_csv(file_path, header=None, prefix='L')
        if not os.path.exists(parent_dir + "single_driver_data"):
            os.mkdir(parent_dir + "single_driver_data")
        for key, value in data.groupby('L0'):
            file_name = key
            # print(file_name)
            fp = open(parent_dir + "single_driver_data\\" + file_name, "a")
            value = value.values
            value_list = value.tolist()
            # write list to file
            for single_list in value_list:
                single_str = str(single_list)
                single_str = single_str[1:-1]
                fp.write(single_str + "\n")
            fp.close()

    # 使用政策表达式匹配所有满足条件的文件
    def find_file_path(self, pattern, parent_dir):
        dir_list = os.listdir(parent_dir)
        dir_str = str(dir_list)
        pattern = re.compile(pattern)
        file_dir_list = pattern.findall(dir_str)
        return file_dir_list


if __name__ == '__main__':
    parent_dir1 = r"E:\\data\\didi\\xian_10\\xian\\"
    pattern = "gps_201610[0-3][0-9]"
    parent_dir2 = r"E:\\data\\didi\\xian_11\\xian\\"
    pattern2 = "gps_201611[0-3][0-9]"
    parent_dir = r"E:\\data\\didi\\xian\\"
    didi = deal_didi()
    didi.deal_all_file(pattern, parent_dir1,pattern2, parent_dir2,parent_dir)
