import os
import glob
import dill
import numpy as np

os.chdir('to_run')
levels = os.listdir()

e = 0
g = 0

for level in levels:
    os.chdir(level)
    frags = glob.glob('*.dill')
    for i in frags:
        #undill and run e, g, hess, apt etc
        infile = open(i, 'rb')
        new_class = dill.load(infile)
        print("Fragment ID:", level, i)
        e += new_class.energy
        g += new_class.grad
    os.chdir('../')

os.chdir('../')
np.save('energy.npy', e)
np.save('gradient.npy', g)
