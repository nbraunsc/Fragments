import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [1.1015, -0.917, 0.5761]], ['H', [1.8827, -1.5674, 0.9847]], ['H', [0.7046013333333333, -1.3504926666666668, -0.35101933333333346]], ['H', [1.5193706666666666, 0.0748186666666667, 0.3600313333333333]], ['H', [0.2859459999999999, -0.8137806666666667, 1.303628666666667]]]
mol.basis = 'sto-3g'
mol.build()
hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
e, g = hf_scanner(mol)
mf = mol.RHF().run()
h = mf.Hessian().kernel()
print('energy no coeff =', e)
print('gradient =', g)
print('hessian =', h)
np.save(os.path.join('8', 'fragment8.pye.npy'), e)
np.save(os.path.join('8', 'fragment8.pyg.npy'), g)
np.save(os.path.join('8', 'fragment8.pyh.npy'), h)
