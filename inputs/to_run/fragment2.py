import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [0.5451, -1.5247, -0.7236]], ['H', [0.1384, -2.5241, -0.5254]], ['H', [1.3532, -1.6488, -1.4549]], ['H', [-0.238568, -0.8767793333333332, -1.1374760000000002]], ['H', [0.9419986666666667, -1.0912073333333332, 0.20351933333333339]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('2', 'fragment2.pye.npy'), e)
np.save(os.path.join('2', 'fragment2.pyg.npy'), g)
np.save(os.path.join('2', 'fragment2.pyh.npy'), h)
