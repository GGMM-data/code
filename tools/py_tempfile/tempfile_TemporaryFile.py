import tempfile


temp = tempfile.TemporaryFile()

try:
    print("create file is: ", temp)
    print("name of temp file is: ", temp.name)
finally:
    print("close the temp file")
    temp.close()
