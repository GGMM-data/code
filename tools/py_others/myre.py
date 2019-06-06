import regex as re


# 缺失id
target1 = "\"LC1\",29,BC,2013"
pattern1 = "^\"[a-zA-Z0-9]*?\",[0-9]*?,[a-zA-Z]*?,[0-9]*?$"

# 缺失name
target2 = "2,29,BC,2013"
pattern2 = "^[0-9]*?,[0-9]*?,[A-Z]*?,[0-9]*?$"

# 缺失age

target3 = '1,"YH1",PG,2013'
pattern3 = "^[0-9]*?,\"[a-zA-Z0-9]*?\",[a-zA-Z]*?,[0-9]*?$"

# 缺失type
target4 = "2,\"LC1\",29,2013"
pattern4 = "^[0-9]*?,\"[a-zA-Z0-9]*?\",[0-9]*?,[0-9]*?$"

# 缺失年份
target5 = "18,\"XY1\",39,BC"
pattern5 = "^[0-9]*?,\"[a-zA-Z0-9]*?\",[0-9]*?,[a-zA-Z]*?$"


print(re.findall(pattern3, target3))
