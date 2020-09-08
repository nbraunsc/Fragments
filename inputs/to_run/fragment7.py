import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [-0.5535, -0.6164, -1.3038]], ['H', [-0.946, -1.0536, -2.2286]], ['H', [-0.12892399999999998, 0.37420600000000015, -1.5122360000000001]], ['H', [0.23016800000000004, -1.2643206666666666, -0.8899239999999999]], ['H', [-1.3622773333333333, -0.5143933333333333, -0.5686386666666665]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('7', 'fragment7.pye.npy'), e)
np.save(os.path.join('7', 'fragment7.pyg.npy'), g)
np.save(os.path.join('7', 'fragment7.pyh.npy'), h)
