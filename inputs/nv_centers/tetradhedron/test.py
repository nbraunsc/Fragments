import pyscf
from pyscf import molden
from pyscf import gto, dft, scf, ci, cc, mp, hessian, lib, grad, mcscf
from pyscf.geomopt.berny_solver import optimize

mol = gto.Mole()
mol.build(
    atom = 'N  0  0  0; N  0  0  2',
    basis = 'ccpvtz',
    symmetry = True,
)
myhf = scf.RHF(mol)
myhf.kernel()

## 6 orbitals, 8 electrons
#state_id : 0 is ground state, state_id=1 would be first
state_id = 0
mc = mcscf.CASSCF(myhf, 8, 4).state_specific_(state_id)
mc.verbose = 5
hartree = mc.kernel()[0]
energy = hartree*27.2114
molden.from_mo(mol, 'cas_ground.molden', mc.mo_coeff)
print("energy in eV = ", energy)

print("mc.kernel ################\n", mc.kernel())
print("mc.mo_coeff ################\n", mc.mo_coeff)
print("mc.mo_energy ################\n", mc.mo_energy)
print("mo energy has len = ", len(mc.mo_energy))
print("trying to print specific one", mc.mo_energy[1])
