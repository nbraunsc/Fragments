import pickle
import dill
import numpy as np
import os
import sys

batch = sys.argv[1] #should be a list of filenames as an argument i.e. python run.py [frag1, frag2, ...]

for i in batch:
    #unpickle and run e, g, hess, apt etc
    infile = open(i, 'rb')
    new_class = pickle.load(infile)
    infile.close()
    new_class.qc_backend()

    #repickle with updated fragment e, g, hess, apt, etc
    outfile = open(i, "wb")
    pickle.dump(new_class, outfile)
    outfile.close()


    
