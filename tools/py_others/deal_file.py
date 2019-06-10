import regex as re
import numpy as np
import argparse

# 检查id
def check_id(e_id):
    if not e_id.isdigit():
        return -1
    return 0


# 检查name
def check_name(e_name):
    if e_name.isdigit():
        return -1
    return 0


# 检查age
def check_age(e_age):
    if not e_age.isdigit():
        return -1
    e_age = int(e_age)
    if e_age > 85 or e_age < 16:
        return -1
    return 0


# 检查type
def check_type(e_type):
    if not e_type.isalpha():
        return -1
    if e_type not in ['PG', 'BC', 'BO']:
        return -1
    return 0


# 检查year
def check_year(e_year):
    if not e_year.isdigit():
        return -1
    if len(e_year) != 4:
        return -1
    return 0


# 检查5个items哪些有问题
# 1,2,3,4,5分别表示id, name, age, type, year有错误，0表示该数据正常
def check_types(items):
    types = []
    # 　check e_id
    if check_id(items[0])!= 0:
        types.append(1)

    # check e_name
    if check_name(items[1]) != 0:
        types.append(2)

    # check e_age
    if check_age(items[2]) != 0:
        types.append(3)

    # check e_type
    if check_type(items[3]) != 0:
        types.append(4)

    # check e_year
    if check_year(items[4]) != 0:
        types.append(5)

    return types


# 检查每条数据
def check_item(item):
    t = 0
    items = item.split(",")
    len_items = len(items)
    if len_items == 4 or len_items == 5:
        # 判断缺失值是哪一个
        if len_items == 4:
            # 缺失id
            pattern1 = "^\"[a-zA-Z0-9]*?\",[0-9]*?,[a-zA-Z]*?,[0-9]*?$"
            # 缺失name
            pattern2 = "^[0-9]*?,[0-9]*?,[A-Z]*?,[0-9]*?$"
            # 缺失age
            pattern3 = "^[0-9]*?,\"[a-zA-Z0-9]*?\",[a-zA-Z]*?,[0-9]*?$"
            # 缺失type
            pattern4 = "^[0-9]*?,\"[a-zA-Z0-9]*?\",[0-9]*?,[0-9]*?$"
            # 缺失年份
            pattern5 = "^[0-9]*?,\"[a-zA-Z0-9]*?\",[0-9]*?,[a-zA-Z]*?$"
            patterns = [pattern1, pattern2, pattern3, pattern4, pattern5]
            for i in range(5):
                if len(re.findall(patterns[i], item)) != 0:
                    items.insert(i, "NULL")
                    break
        if len(items) == 4:
            t = -2
            types = []
            print("Data is illegal!")
            return (t, types, items)


        types = check_types(items)
        if len(types) != 0:
            t = -1

    else:
        t = -2
        types = []
        print("Data is illegal!")
    return (t, types, items)


def list2str(l):
    temp_str = ""
    for item in l:
        temp_str += str(item) + ","
    temp_str = temp_str[:-1]
    return temp_str+"\n"


if __name__ == "__main__":
    # 要处理的文件名
    parse = argparse.ArgumentParser(add_help="input and output file name")
    parse.add_argument("--input_filename", default="employee.txt", type=str, help="--input filename")
    parse.add_argument("--output_filename", default="correct_employee.csv", type=str, help="--output filename")
    args = parse.parse_args()

    with open(args.input_filename, "r") as f:
        # 记录正确的data
        correct_list = []
        # 记录错误的data
        error_list = []

        # 存储正确数据的信息（age, type, year)
        e_type = []
        e_age = []
        e_year = []

        # 处理每一条数据
        for line in f:
            line = line.replace("\n","")
            # 检查每一行data
            (t, types, items) = check_item(line)
            if t == -2:
                continue
            # t=0表示该条数据正确
            if t == 0:
                # 记录该条正确数据
                correct_list.append(items)
                e_age.append(int(items[2]))
                e_type.append(items[3])
                e_year.append(int(items[4]))
            # 否则就说明有错，需要校正
            else:
                error_list.append([items, types])

        # id
        max_id = int(correct_list[len(correct_list)-1][0])
        id = max_id + 1
        # name
        name_count = 1
        # 计算age的平均数和中位数
        e_age_mean = int(np.mean(e_age))
        e_age_median = int(np.median(e_age))
        # 计算雇员type的众数
        e_types = ['PG', 'BC', 'BO']
        e_type_counts = [e_type.count(x) for x in e_types]
        type_max = np.argmax(e_type_counts)
        e_type_mode = e_types[type_max]
        # year的众数
        years = list(set(e_year))
        e_year_counts = [e_year.count(x) for x in years]
        year_max = np.argmax(e_year_counts)
        e_year_mode = years[year_max]

        # 对错误数据进行校正
        after_correct = []
        for items, errors in error_list:
            for e in errors:
                if e == 1: # id
                    items[0] = id
                    id += 1
                    continue
                if e == 2: # namee
                    items[1] = "\"ename_" + str(name_count) + "\""
                    name_count += 1
                    continue
                if e == 3: # age
                    items[2] = e_age_mean
                    continue
                if e == 4: # type
                    items[3] = e_type_mode
                    continue
                if e == 5: # year
                    items[4] = e_year_mode
                    continue
            # items[1] = items[1].replace("\"","")
            after_correct.append(items)

        # 将校正后的数据存起来
        ff = open(args.output_filename, "w")
        for items in correct_list:
            ff.write(list2str(items))
        for items in after_correct:
            ff.write(list2str(items))
        ff.close()