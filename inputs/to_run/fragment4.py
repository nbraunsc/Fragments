import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [-0.0418, -0.7723, 1.596]], ['H', [-0.4584, -1.7588, 1.8337]], ['H', [0.3441, -0.355, 2.5341]], ['H', [0.773754, -0.8755193333333333, 0.8684713333333332]], ['H', [-0.8276793333333332, -0.12131199999999998, 1.1915399999999998]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('4', 'fragment4.pye.npy'), e)
np.save(os.path.join('4', 'fragment4.pyg.npy'), g)
np.save(os.path.join('4', 'fragment4.pyh.npy'), h)
