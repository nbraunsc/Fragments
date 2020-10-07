import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
len_prim =5
num_la =2
step_size = 0.001
mol = gto.Mole()
mol.atom = [['C', [1.53371, -0.42208, 0.00795]], ['C', [0.48841, 0.65027, -0.23351]], ['O', [2.71181, -0.09054, 0.12791]], ['H', [0.73137, 1.49653, 0.42043]], ['H', [0.5847, 0.98117, -1.26925]], ['H', [1.2241019333333334, -1.4645096666666666, -0.015575733333333336]], ['H', [-0.5409157333333334, 0.3552781333333333, -0.04314273333333332]]]
mol.basis = 'sto-3g'
mol.build()
mp2_scanner = mp.MP2(scf.RHF(mol)).nuc_grad_method().as_scanner()
e, g = mp2_scanner(mol) 
h = 0
#If not analytical hess, not do numerical below
if type(h) is int:
    hess = np.zeros(((len(mol.atom))*3, (len(mol.atom))*3))
    i = -1
    for atom in range(0, len(mol.atom)):
        for xyz in range(0, 3):
            i = i+1
            mol.atom[atom][1][xyz] = mol.atom[atom][1][xyz]+step_size
            mol2 = gto.Mole()
            mol2.atom = mol.atom
            mol2.basis = mol.basis
            mol2.build()
            mp2_scanner = mp.MP2(scf.RHF(mol2)).nuc_grad_method().as_scanner()
            e, g = mp2_scanner(mol2)
            grad1 = g.flatten()
            mol.atom[atom][1][xyz] = mol.atom[atom][1][xyz]-2*step_size
            mol2 = gto.Mole()
            mol2.atom = mol.atom
            mol2.basis = mol.basis
            mol2.build()
            mp2_scanner = mp.MP2(scf.RHF(mol2)).nuc_grad_method().as_scanner()
            e, g = mp2_scanner(mol2)
            grad2 = g.flatten()
            mol.atom[atom][1][xyz] = mol.atom[atom][1][xyz]+step_size
            vec = (grad1 - grad2)/(4*step_size)
            hess[i] = vec
            hess[:,i] = vec
    h = hess.reshape(len_prim+num_la, 3, len_prim+num_la, 3)
    h = h.transpose(0, 2, 1, 3)
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('5', 'fragment5.pye.npy'), e)
np.save(os.path.join('5', 'fragment5.pyg.npy'), g)
np.save(os.path.join('5', 'fragment5.pyh.npy'), h)
