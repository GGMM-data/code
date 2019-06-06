import re

text = """First line.
Second line.
Third line."""

pattern = "^.*$"  # 匹配从开始到结束的任何字符
# 默认情况下， . 不匹配newlines，所以默认情况下不会有任何匹配结果，因为$之前有newline，而.不能匹配
# re.search(pattern, text) is None  # Nothing matches!
print(re.search(pattern, text))

# 如果设置MULTILINE模式, $匹配每一行的结尾，这个时候第一行就满足要求了，设置MULTILINE模式后，$匹配string的结尾和每一行的结尾（each newline之前)
print(re.search(pattern, text, re.M).group())
# First line.

# 如果同时设置MULTILINE和DOTALL模式, .能够匹配newlines，所以第一行和第二行的newline都匹配了，在贪婪模式下，就匹配了整个字符串。
print(re.search(pattern, text, re.M | re.S).group())
# First line.
# Second line.
# Third line.

