import pickle
import os
import glob

os.chdir('to_run')
#levels = os.listdir()

e = 0
g = 0
h = 0
apt = 0

for level in os.listdir():
    os.chdir(level)
    frags = glob.glob('*.pickle')
    for i in frags:
        #unpickle and run e, g, hess, apt etc
        infile = open(i, 'rb')
        new_class = pickle.load(infile)
        e += new_class.energy
        g += new_class.grad
        h += new_class.hessian
        apt += new_class.apt 
    os.chdir('../')

print("MIM Energy:", e, "Hartree")
print("MIM Gradient:\n", g)
print("MIM Hessian shape:", h.shape)
print("MIM mass-weighted APT's shape:", apt.shape)
