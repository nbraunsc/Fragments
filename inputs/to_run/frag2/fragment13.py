import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
len_prim =11
num_la =1
step_size = 0.001
mol = gto.Mole()
mol.atom = [['C', [1.09968, -1.88343, -0.02503]], ['H', [0.17726, -1.99469, 0.55416]], ['C', [2.11284, -2.81406, 0.57024]], ['H', [0.90494, -2.14985, -1.06835]], ['C', [1.91502, -4.13708, 0.65078]], ['C', [2.91707, -5.055, 1.25397]], ['H', [3.86663, -5.00121, 0.71071]], ['H', [2.55931, -6.08442, 1.20792]], ['H', [3.08739, -4.80241, 2.30624]], ['H', [3.03761, -2.39017, 0.96112]], ['H', [0.99714, -4.5767, 0.26801]], ['H', [1.4092880666666665, -0.8410003333333333, -0.0015042666666666635]]]
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
            mol2 = gto.Mole()
            mol2.atom = mol.atom
            mol2.basis = mol.basis
            mol2.build()
            hf_scanner = scf.RHF(mol2).apply(grad.RHF).as_scanner()
            e, g = hf_scanner(mol2)
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
np.save(os.path.join('13', 'fragment13.pye.npy'), e)
np.save(os.path.join('13', 'fragment13.pyg.npy'), g)
np.save(os.path.join('13', 'fragment13.pyh.npy'), h)
