import numpy as np
import os


os.path.abspath(os.curdir)
os.chdir('to_run/')
#files = os.listdir()

numpy_list = [x[0] for x in os.walk('.')]
numpy_list.pop(0)
print("full list with os.walk", numpy_list)

energy = 0
grad = 0
hess = 0
for i in numpy_list:
    print(i)
    number = i.replace('./', '')
    print("pulling out just number", number, type(number))
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

print("Global energy :", energy)
print("Global gradient: \n", grad)
print("Global hessian: \n", hess)

        
