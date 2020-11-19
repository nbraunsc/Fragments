import pickle
import dill
import numpy as np
import os
import sys
#import nicolefragment
#from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf

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
    print(infile)
    new_class = dill.load(infile)
    print("hello")
    infile.close()
    new_class.qc_backend()
    
    #update status of calculation
    status_name = i.replace(".dill", ".status")
    stat_file = open(status_name, "rb") 
    status = dill.load(stat_file)
    stat_file.close()
   
    #redill .status file with updated status
    stat_file = open(status_name, "wb") 
    dill.dump(status, stat_file)
    stat_file.close()

    #redill with updated fragment e, g, hess, apt, etc
    file_name = i.replace(".status", ".dill")
    outfile = open(file_name, "wb")
    dill.dump(new_class, outfile)
    outfile.close()
