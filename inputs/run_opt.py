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
for i in batch_list:
    #undill and run e, g, hess, apt etc
    infile = open(i, 'rb')
    new_class = dill.load(infile)
    outfile = open(i, "wb")
    e, g, h = new_class.qc_backend()
    new_class.hess_apt(h)

    ##redill with updated fragment e, g, hess, apt, etc
    dill.dump(new_class, outfile)
    infile.close()
    outfile.close()
