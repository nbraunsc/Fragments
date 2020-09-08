import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [0.5954, 1.3931, -0.3013]], ['H', [1.0176, 2.3812, -0.515]], ['H', [0.2004273333333333, 0.9502626666666667, -1.2248526666666668]], ['H', [1.3742886666666667, 0.7370473333333333, 0.10851]], ['H', [-0.21808533333333335, 1.4869746666666666, 0.4297953333333334]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('9', 'fragment9.pye.npy'), e)
np.save(os.path.join('9', 'fragment9.pyg.npy'), g)
np.save(os.path.join('9', 'fragment9.pyh.npy'), h)
