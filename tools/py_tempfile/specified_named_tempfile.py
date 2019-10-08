import tempfile

temp = tempfile.NamedTemporaryFile(prefix="prefix", suffix="suffix")

try:
    print("create file is: ", temp)
    print("name of temp file is: ", temp.name)
finally:
    print("close the temp file")
    temp.close()
