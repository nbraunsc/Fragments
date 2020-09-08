import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [-1.1435, 0.1403, 1.029]], ['H', [-1.9545, 0.2398, 1.7589]], ['H', [-1.5314106666666667, -0.2974726666666667, 0.10009733333333315]], ['H', [-0.7165699999999999, 1.1278386666666667, 0.811148]], ['H', [-0.3576206666666667, -0.510688, 1.4334600000000002]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('3', 'fragment3.pye.npy'), e)
np.save(os.path.join('3', 'fragment3.pyg.npy'), g)
np.save(os.path.join('3', 'fragment3.pyh.npy'), h)
