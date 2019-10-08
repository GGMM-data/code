import tempfile

temp = tempfile.TemporaryFile()

try:
    temp.write(b"hello world")
    temp.seek(0)
    print(temp.read())

finally:
    temp.close()
