import regex as re


match1 = re.match("cd", "abcdef")     # match
match2 = re.search("cd", "abcdef")    # search
print(match1)
print(match2)
print(match2.group(0))


with open("content.txt", "r") as f:
    s = f.read()
match3 = re.match("cd", s)     # match
match4 = re.search("cd", s)
print(match3)
print(match4)

