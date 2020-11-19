import dill
import os
import glob


os.path.abspath(os.curdir)
os.chdir('to_run/')
status_list = []
status = 'false'

def opt_fnc():
    for i in os.listdir():
        os.chdir(i)
        files = glob.glob('*.status')
        for j in files:
            infile = open(j, 'rb')
            var = dill.load(infile)
            status_list.append(var)
            infile.close()
        os.chdir('../')

    if 0 or -1 in status_list:
        status = 'false'

    else:
        status = 'true'
    
    return status


if __name__ == '__main__':
    opt_fnc()

