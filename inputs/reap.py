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
    print(frags)
    for i in frags:
        #unpickle and run e, g, hess, apt etc
        infile = open(i, 'rb')
        new_class = pickle.load(infile)
        print("energy:", new_class.energy)
        print("grad:", new_class.grad)
        print("hess type and shape:", type(new_class.hessian), new_class.hessian.shape)
        e += new_class.energy
        g += new_class.grad
        h += new_class.hessian
        apt += new_class.apt 
    os.chdir('../')

print(e)
print(g)
print(h.shape)
print(apt.shape)
