import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [1.6873, 0.4734, 0.2732]], ['H', [2.5149, 0.3834, -0.4411]], ['H', [2.1028, 0.912, 1.1887]], ['H', [0.9084113333333333, 1.1294526666666667, -0.13661]], ['H', [1.2694293333333333, -0.5184186666666668, 0.48926866666666663]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('5', 'fragment5.pye.npy'), e)
np.save(os.path.join('5', 'fragment5.pyg.npy'), g)
np.save(os.path.join('5', 'fragment5.pyh.npy'), h)
