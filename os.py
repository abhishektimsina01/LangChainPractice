import os

# it give us the current directory path 
print(os.getcwd())

# list out all the files and directories in the current directory (cwd)
files = os.listdir()
print(files)
print("test.py" in files)


# make and remove the directories
if "src" in os.listdir():
    os.rmdir("src")
else:
    os.mkdir("src")

# check if the file exist or not
print(os.path.exists(".env"))
print(os.path.exists(".env1"))

print(os.path.isfile("RAG.py"))
print(os.path.isdir("venv"))

path = os.getcwd()
print(os.path.basename(path))
print(os.path.dirname(path))
print(os.path.splitext(path))