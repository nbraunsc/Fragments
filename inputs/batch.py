#python batch.py <batch size>

import os
import sys
import glob

batch_size = int(sys.argv[1])

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

os.path.abspath(os.curdir)
os.chdir('to_run/')
for i in os.listdir():
    os.chdir(i)
    files_pickle = glob.glob('*.pickle')
    if batch_size == None:
        #bash_command = "qsub pbs.sh"
        bash_command = "python run.py" + str(list(files_pickle))
        os.chdir('../../')
        os.system(bash_command)
    else:
        for x in batch(files_pickle, 3):
            os.chdir('../../')
            #cmd = 'python run.py %s %s'%(i, x)
            cmd = 'qsub pbs.sh %s %s'%(i, x)
            os.system(cmd)
            os.chdir('to_run/'+ i)
    os.chdir('../')
