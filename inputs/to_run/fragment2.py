import numpy as np

import os

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf

from pyscf.prop.freq import rhf

from pyscf.prop.polarizability import rhf

from pyscf.grad.rhf import GradientsBasics

np.set_printoptions(suppress=True, precision=5)
mol = gto.Mole()
mol.atom = [['C', [1.09968, -1.88343, -0.02503]], ['H', [0.17726, -1.99469, 0.55416]], ['C', [2.11284, -2.81406, 0.57024]], ['H', [0.90494, -2.14985, -1.06835]], ['C', [1.91502, -4.13708, 0.65078]], ['H', [3.03761, -2.39017, 0.96112]], ['H', [0.99714, -4.5767, 0.26801]], ['H', [1.4092880666666665, -0.8410003333333333, -0.0015042666666666635]], ['H', [2.6298156666666666, -4.791862933333333, 1.0810555333333334]]]
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
