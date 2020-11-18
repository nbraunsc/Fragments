import pickle
#import dill
import numpy as np
import os
import sys
import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf

directory = sys.argv[1] #should be the directory of mim level
batch = sys.argv[2].split("_")
batch_list = []
for i in batch:
    if not i:
        continue
    else:
        entry = "fragment" + i + ".pickle"
        batch_list.append(entry)

print("Batch list:", batch_list)
os.chdir('to_run/')
os.chdir(directory)
for i in batch_list:
    #unpickle and run e, g, hess, apt etc
    infile = open(i, 'rb')
    new_class = pickle.load(infile)
    print("hello")
    infile.close()
    new_class.qc_backend()
    
    #update status of calculation
    status_name = i.replace(".pickle", ".status")
    stat_file = open(status_name, "rb") 
    status = pickle.load(stat_file)
    stat_file.close()
    if new_class.energy == None or new_class.grad == None or new_class.hessian == None:
        status = -1
    else:
        status = 1
    
    stat_file = open(status_name, "wb") 
    pickle.dump(status, stat_file)
    stat_file.close()

    #repickle with updated fragment e, g, hess, apt, etc
    file_name = i.replace(".status", ".pickle")
    outfile = open(file_name, "wb")
    pickle.dump(new_class, outfile)
    outfile.close()
