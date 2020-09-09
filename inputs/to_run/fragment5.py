import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [1.53371, -0.42208, 0.00795]], ['C', [0.48841, 0.65027, -0.23351]], ['O', [2.71181, -0.09054, 0.12791]], ['H', [0.73137, 1.49653, 0.42043]], ['H', [0.5847, 0.98117, -1.26925]], ['H', [1.2241019333333334, -1.4645096666666666, -0.015575733333333336]], ['H', [-0.5409157333333334, 0.3552781333333333, -0.04314273333333332]]]
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
