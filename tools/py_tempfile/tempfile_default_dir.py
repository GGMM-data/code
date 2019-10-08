import tempfile


print("current temp dir: ", tempfile.gettempdir())

tempfile.tempdir = "/temp"

print("current temp dir: ", tempfile.gettempdir())

