import os

def find_new_file(dir):
    '''查找目录下最新的文件'''
    file_list = []
    for f in os.listdir(dir):
        if os.path.isdir(f):
            file_list.append(f)
    file_list.sort(key=lambda fn: os.path.getmtime(dir + "/" + fn))
    if len(file_list) > 0:
        return file_list[-1]
    else:
        return None

if __name__ == "__main__":
    print(find_new_file("."))
