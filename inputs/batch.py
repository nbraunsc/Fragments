import os
import sys
import glob
import nicolefragment
from nicolefragment import *
#from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf

batch_size = int(sys.argv[1])
print(batch_size)
script = sys.argv[2]
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

os.path.abspath(os.curdir)
os.chdir('to_run/')
command_list = []
for i in os.listdir():
    os.chdir(i)
    files_dill = glob.glob('*.dill')
    if batch_size == None:
        #bash_command = "qsub pbs.sh"
        cmd = 'python run_opt.py %s %s'%(path, string_num)
        os.chdir('../../')
        os.system(cmd)
    else:
        for x in batch(files_dill, batch_size):
            print("hello")
            path = os.getcwd()
            string_num = str()
            for frag in x:
                y = frag.replace("fragment", "").replace(".dill", "")
                string_num+=y+"_"
            submit_name = i + "_" + string_num
            os.chdir('../../')
            cmd = 'python run_opt.py %s %s'%(path, string_num)
            #cmd = 'qsub -N %s -v LEVEL="%s",BATCH="%s" %s'%(submit_name, path, string_num, script)
            command_list.append(cmd)
            #os.system(cmd)
            os.chdir('to_run/')
            os.chdir(i)
            #print("Path to after command:", os.getcwd(), "should be frag1 or frag2...")
    os.chdir('../')
os.chdir('../')

for command in command_list:
    os.system(command)
    print("submitting job:", command)


