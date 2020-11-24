import pickle
import os
import glob
import dill
import numpy as np

os.chdir('to_run')
levels = os.listdir()

e = 0
g = 0
h = 0
#apt = 0

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
        #h += new_class.hessian
        #apt += new_class.apt 
    os.chdir('../')

os.chdir('../')
np.save('energy.npy', e)
np.save('gradient.npy', g)
#np.save('hessian.npy', h)

print("MIM Energy:", e, "Hartree")
print("MIM Gradient:\n", g)
#print("MIM Hessian shape:", h.shape)
#print("MIM mass-weighted APT's shape:", apt.shape)
