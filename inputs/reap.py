import pickle
import os

os.chdir('to_run')
levels = os.listdir()

e = 0
g = 0
h = 0
apt = 0

for level in levels:
    os.chdir(level)
    frags = os.listdir()
    for i in frags:
        #unpickle and run e, g, hess, apt etc
        infile = open(i, 'rb')
        new_class = pickle.load(infile)
        e += new_class.energy
        g += new_class.grad
        h += new_class.hessian
        apt += new_class.apt 
    os.chdir('../')

print(e)
print(g)
print(h.shape)
print(apt.shape)
