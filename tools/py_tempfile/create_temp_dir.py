import tempfile
import os


# temp_dir = tempfile.TemporaryDirectory(prefix="prefix_")
temp_dir = tempfile.TemporaryDirectory()
print("Directory name: ", temp_dir.name)

os.removedirs(temp_dir.name)
