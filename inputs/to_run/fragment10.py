import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [0.0417, 0.7723, -1.596]], ['H', [0.8413, 0.6875, -2.3422]], ['H', [-0.727, 1.4263, -2.0258]], ['H', [-0.382876, -0.2183060000000001, -1.387564]], ['H', [0.43667266666666676, 1.2151373333333333, -0.6724473333333333]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('10', 'fragment10.pye.npy'), e)
np.save(os.path.join('10', 'fragment10.pyg.npy'), g)
np.save(os.path.join('10', 'fragment10.pyh.npy'), h)
