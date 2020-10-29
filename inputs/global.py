import numpy as np
import os


os.path.abspath(os.curdir)
os.chdir('to_run/')

#numpy_list = [x[0] for x in os.walk('.')]
#numpy_list.pop(0)

#energy = 0
#grad = 0
#hess = 0

files = os.listdir()
global_e = 0
global_g = 0
global_h = 0
for level in range(0, len(files)):
    os.chdir(files[level])
    mim_coeff = np.load('mim_co.npy')
    numpy_list = [x[0] for x in os.walk('.')]
    numpy_list.pop(0)
    energy = 0
    grad = 0
    hess = 0
    for i in numpy_list:
        number = i.replace('./', '')
        os.path.abspath(os.curdir)
        os.chdir(number)
        coeff = np.load('coeff' + number + '.npy')
        j_grad = np.load('jacob_grad' + number + '.npy')
        j_hess = np.load('jacob_hess' + number + '.npy')
        e = np.load('fragment'+ number + '.pye.npy')
        g = np.load('fragment'+ number + '.pyg.npy')
        h = np.load('fragment'+ number + '.pyh.npy')
        j_hess_t = j_hess.transpose(1,0,2,3)
        y = np.einsum('ijkl, jmln -> imkn', j_hess, h) 
        energy += e*coeff
        grad += coeff*j_grad.dot(g)
        hess += np.einsum('ijkl, jmln -> imkn', y, j_hess_t)*coeff
        os.chdir('../')
    global_e += energy*mim_coeff
    global_g += grad*mim_coeff
    global_h += hess*mim_coeff
    os.chdir('../')

print("Global energy :", global_e)
print("Global gradient: \n", global_g)
print("Global hessian shpae: \n", global_h.shape)

        
