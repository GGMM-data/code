import subprocess

print("$ nslookup www.baidu.com")
r = subprocess.call(["nslookup","baidu.com"])
print("Exit code:", r)

p = subprocess.Popen(["nslookup"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
output, err = p.communicate(b"www.baidu.com")
print(output.decode("utf-8"))
print("Exit code:", p.returncode)
