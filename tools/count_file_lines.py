def file_len(file_name):
    with open(file_name) as f:
        for i, l in enumerate(f):
            pass
    return i+1

f = open(file_name, "r")
lines = sum(1 for _ in f)
