import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [-0.545, 1.5247, 0.7236]], ['H', [-0.1677, 1.9815, 1.6469]], ['H', [-1.3238, 2.1915, 0.3335]], ['H', [0.26848533333333335, 1.4308253333333334, -0.007495333333333409]], ['H', [-0.97193, 0.5371613333333334, 0.941452]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('1', 'fragment1.pye.npy'), e)
np.save(os.path.join('1', 'fragment1.pyg.npy'), g)
np.save(os.path.join('1', 'fragment1.pyh.npy'), h)
