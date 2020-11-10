import pickle
import dill
import numpy as np
import os
import sys

directory = sys.argv[1] #should be the directory of mim level
batch = sys.argv[2:] #should be a list of filenames as an argument i.e. python run.py [fragment0.pickle, fragment1.pickle, ...]
batch_list = []
for file_name in batch:
    filename = file_name
    if '[' in file_name:
        print('[')
        filename = filename.replace('[', "")
    if ']' in file_name:
        filename = filename.replace(']', "")
        print(']')
    if ',' in file_name:
        filename = filename.replace(',', "")
        print(',')
    batch_list.append(filename)

os.chdir('to_run/' + directory)
for i in batch_list:
    #unpickle and run e, g, hess, apt etc
    infile = open(i, 'rb')
    new_class = pickle.load(infile)
    infile.close()
    new_class.qc_backend()
    
    #update status of calculation
    status_name = i + ".status"
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
    outfile = open(i, "wb")
    pickle.dump(new_class, outfile)
    outfile.close()
