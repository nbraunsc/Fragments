import pickle
import dill
import numpy as np
import os
import sys

directory = sys.argv[1] #should be the directory of mim level
batch = sys.argv[2].split("_")
batch_list = []
for i in batch:
    if not i:
        continue
    else:
        entry = "fragment" + i + ".dill"
        batch_list.append(entry)

print("Batch list:", batch_list)
os.chdir('to_run/')
os.chdir(directory)
status = 0
for i in batch_list:
    #undill and run e, g, hess, apt etc
    infile = open(i, 'rb')
    new_class = dill.load(infile)
    outfile = open(i, "wb")
    new_class.qc_backend()

    if new_class.energy == 0:
        status = -1
    if type(new_class.grad) is int:
        status = -1
    #if type(new_class.hessian) is int:
    #    status = -1
    else:
        status = 1

    ##redill with updated fragment e, g, hess, apt, etc
    dill.dump(new_class, outfile)
    infile.close()
    outfile.close()
    
    #update status of calculation
    status_name = i.replace(".dill", ".status")
    out_stat = open(status_name, "wb")
    dill.dump(status, out_stat)
    out_stat.close()
