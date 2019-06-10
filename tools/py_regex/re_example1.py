import regex as re

# 找出每一行的数字
string = """a9apple1234
2banana5678
a3coconut9012"""
pattern = "[0-9]+"

## search
result = re.search(pattern, string)
print(type(result))
print(result[0])
print(result.group(0))

## match
# 即使设置了MULTILINE模式，也只会匹配string的开头而不是每一行的开头
result = re.match(pattern, string, re.S| re.M)  
print(type(result))
# print(result[0])
# print(result.group(0))

## findall
result = re.findall(pattern, string)
print(type(result))
print(result)

