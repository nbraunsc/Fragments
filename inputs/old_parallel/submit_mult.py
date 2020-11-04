import os
import glob

os.getcwd()
os.chdir('to_run/')

levels = os.listdir(os.getcwd())
print("levels", levels)
for i in levels:
    print(i)
    os.chdir(i)
    for file in glob.glob("*.py"):
        print("file to be run:", file, type(file))
        command = "python" + " " + file
        print(command)
        os.system(command)
    os.chdir('../')
