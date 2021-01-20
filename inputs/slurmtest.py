import pyscf
from pyscf import gto, dft, scf, ci, cc, mp, hessian, lib, grad, mcscf
from pyscf.geomopt.berny_solver import optimize

mol = gto.Mole()
mol.atom = '''
H 0 0.761811000000000016 -0.598709999999999964
O 0 0 0
H 0 -0.761811000000000016 -0.598709999999999964
'''

mol.basis = '6-31g*'
mol.build(symmetry=True)
mf = scf.RHF(mol)
mf.kernel()
print(mf.kernell())
print(mol.atom_coords())
