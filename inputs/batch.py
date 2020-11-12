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
#print("Path at start:", os.getcwd(), "should be in to_run")
for i in os.listdir():
    os.chdir(i)
    #print("Path at level:", os.getcwd(), "should be in frag1 or frag2..")
    files_pickle = glob.glob('*.pickle')
    if batch_size == None:
        #print("no batch_size")
        #bash_command = "qsub pbs.sh"
        bash_command = "python run.py" + str(list(files_pickle))
        os.chdir('../../')
        #print("Path to do command:", os.getcwd(), "should be inputs directory")
        os.system(bash_command)
    else:
        for x in batch(files_pickle, 3):
            path = os.getcwd()
            string_num = str()
            for frag in x:
                y = frag.replace("fragment", "").replace(".pickle", "")
                string_num+=y+"_"
            submit_name = i + "_" + string_num
            os.chdir('../../')
            #cmd = 'python run.py %s %s'%(i, x)
            cmd = 'qsub -N %s -v LEVEL="%s",BATCH="%s" pbs.sh'%(submit_name, path, string_num)
            print(cmd)
            os.system(cmd)
            os.chdir('to_run/')
            os.chdir(i)
            #print("Path to after command:", os.getcwd(), "should be frag1 or frag2...")
    os.chdir('../')
    #print("Path for level for loop:", os.getcwd(), "should be next level frag2..")
os.chdir('../')
#print("Path at end of loop:", os.getcwd(), "should be inputs")
