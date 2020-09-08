import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [-1.6873, -0.4734, -0.2732]], ['H', [-2.4857, 0.1592, -0.6803]], ['H', [-2.1321, -1.4548, -0.0673]], ['H', [-0.8785226666666667, -0.5754066666666666, -1.0083613333333337]], ['H', [-1.2993893333333333, -0.03562733333333329, 0.6557026666666668]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('6', 'fragment6.pye.npy'), e)
np.save(os.path.join('6', 'fragment6.pyg.npy'), g)
np.save(os.path.join('6', 'fragment6.pyh.npy'), h)
