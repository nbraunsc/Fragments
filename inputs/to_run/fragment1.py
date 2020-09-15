import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
len_prim =9
num_la =2
step_size = 0.001
mol = gto.Mole()
mol.atom = [['C', [1.53371, -0.42208, 0.00795]], ['O', [2.71181, -0.09054, 0.12791]], ['C', [1.09968, -1.88343, -0.02503]], ['H', [0.17726, -1.99469, 0.55416]], ['C', [2.11284, -2.81406, 0.57024]], ['H', [0.90494, -2.14985, -1.06835]], ['C', [1.91502, -4.13708, 0.65078]], ['H', [3.03761, -2.39017, 0.96112]], ['H', [0.99714, -4.5767, 0.26801]], ['H', [0.7880626666666666, 0.34286300000000014, -0.16429146666666666]], ['H', [2.6298156666666666, -4.791862933333333, 1.0810555333333334]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()

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
            hf_scanner = scf.RHF(mol2).apply(grad.RHF).as_scanner()
            e, g = hf_scanner(mol2)
            grad1 = g.flatten()
            mol.atom[atom][1][xyz] = mol.atom[atom][1][xyz]-2*step_size
            mol3 = gto.Mole()
            mol3.atom = mol.atom
            mol3.basis = mol.basis
            mol3.build()
            hf_scanner = scf.RHF(mol3).apply(grad.RHF).as_scanner()
            e2, g2 = hf_scanner(mol3)
            grad2 = g2.flatten()
            mol.atom[atom][1][xyz] = mol.atom[atom][1][xyz]+step_size
            vec = (grad1 - grad2)/(4*step_size)
            hess[i] = vec
            hess[:,i] = vec
    h = hess.reshape(len_prim+num_la, 3, len_prim+num_la, 3)
    h = hess.transpose(0, 2, 1, 3)

print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('1', 'fragment1.pye.npy'), e)
np.save(os.path.join('1', 'fragment1.pyg.npy'), g)
np.save(os.path.join('1', 'fragment1.pyh.npy'), h)
