### Don't need this python script for MIM code ###

import numpy as np
from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf
from pyscf.prop.freq import rhf
from pyscf.prop.polarizability import rhf
from pyscf.grad.rhf import GradientsBasics
from pyscf.geomopt.berny_solver import optimize

np.set_printoptions(suppress=True, precision=5)

#input_xyz = [['O', [1.000000, 0.000000, 0.1519867318]], ['H', [-0.7631844707, 0.000000, -0.4446133658]], ['H', [0.7631844707, 0.000000, -0.4446133658]]]
input_xyz = [['O', [1.000000, 0.000000, 0.1519867318]], ['H', [-0.7631844707, 0.000000, -0.4446133658]], ['H', [0.7631844707, 0.000000, -0.4446133658]]]
coord_matrix = np.zeros((3,3))
x = [1.000000, 0.000000, 0.1519867318]
y = [-0.7631844707, 0.000000, -0.4446133658]
z = [0.7631844707, 0.000000, -0.4446133658]
coord_matrix[0] = x
coord_matrix[1] = y
coord_matrix[2] = z
#coords_list = [1.000000, 0.000000, 0.1519867318, -0.7631844707, 0.000000, -0.4446133658, 0.7631844707, 0.000000, -0.4446133658]

mol = gto.Mole()
mol.atom = input_xyz
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g_org = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
g_flat = g_org.flatten()

step = 0.001
def gradient(step, xyz):
    mol = gto.Mole()
    mol.atom = xyz
    mol.basis = 'sto-3g'
    mol.build()
    hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
    e, g = hf_scanner(mol)
    py_grad = g.flatten()
    return py_grad

#for atomi in range(0, 3):   #atom iteration
#    for atomj in range(0, 3):
#        for comp in range(0, 3): #xyz iteration
#            #input_xyz[atomj][1][comp] = input_xyz[atomj][1][comp] + step
#            g_org[atomj][comp] = g_org[atomj][comp] + step
#            grad1 = gradient(step, input_xyz)
#            #input_xyz[atomj][1][comp] = input_xyz[atomj][1][comp] - 2*step
#            grad2 = gradient(step, input_xyz)
#            input_xyz[atomj][1][comp] = input_xyz[atomj][1][comp] + step
#            
#            #input_xyz[atomi][1][comp] = input_xyz[atomi][1][comp] + step
#            grad3 = gradient(step, input_xyz)
#            #input_xyz[atomi][1][comp] = input_xyz[atomi][1][comp] - 2*step
#            grad4 = gradient(step, input_xyz)
#            #input_xyz[atomi][1][comp] = input_xyz[atomi][1][comp] + step
#            vec = (grad1[atomi] - grad2[atomi] + grad3[atomj] - grad4[atomj])/(4*step**2)
#            #vec = (grad1[atomj] - grad2[atomj])/(4*step) + (grad3[atomi]-grad4[atomi])/(4*step)
#            print("VEC", vec)
#            vector = vec.sum(axis=0)
#            print("vector", vector)
#            hessian[atomi][atomj][comp] = vec[atomi]

#""" this one is sorta working """
#print("pyscf hess \n", h)
#hessian = np.zeros((3,3,3,3))
#for atomi in range(0, 3):   #atomi iterations (orginial gradient is w.r.t atomi)
#    for index in range(0, 3):   #atomj iterations
#        for cart in range(0, 3):    #xyz iterations
#            input_xyz[index][1][cart] = input_xyz[index][1][cart]+step
#            print(input_xyz)
#            grad1 = gradient(step, input_xyz)
#            input_xyz[index][1][cart] = input_xyz[index][1][cart]-step
#            vec = (grad1[index] - g_org[index])/(2*step)
#            print("vector", vec)
#            hessian[atomi][index][cart] = vec
#            hessian[atomi][index][:,cart] = vec
#            print("hessian block")
#            print(hessian[index])
#

#hess_num = hessian.transpose(0, 2, 1, 3)
#hess_num_flat = hess_num.reshape(hess.shape[0]*hess.shape[1], hess.shape[2]*hess.shape[3])
#print("pyscf hess \n", hess_num_flat)

#print("pyscf hess \n", h)
#print("my hessian", "\n", hess_num_flat)
#print("diff", "\n", h-hessian)

"""attempt number 3, central difference"""
hessian = np.zeros((9,9))
matrix = coord_matrix.flatten()
for atom in range(0, len(matrix)):
    matrix[atom] = matrix[atom] + step
    input_xyz = [['O', [matrix[0], matrix[1], matrix[2]]], ['H', [matrix[3], matrix[4], matrix[5]]], ['H', [matrix[6], matrix[7], matrix[8]]]]
    grad1 = gradient(step, input_xyz)
    matrix[atom] = matrix[atom] - 2*step
    input_xyz = [['O', [matrix[0], matrix[1], matrix[2]]], ['H', [matrix[3], matrix[4], matrix[5]]], ['H', [matrix[6], matrix[7], matrix[8]]]]
    grad2 = gradient(step, input_xyz)
    vec = (grad1 - grad2)/(4*step)
    #vec = (grad1 - g_flat)/(2*step)
    print(vec.shape)
    hessian[atom] = vec
    hessian[:,atom] = vec

hess = h.transpose(0, 2, 1, 3)
hess_flat = hess.reshape(hess.shape[0]*hess.shape[1], hess.shape[2]*hess.shape[3])
print("pyscf hess \n", hess_flat)
print("my hessian", hessian)
print("difference", hess_flat - hessian)

