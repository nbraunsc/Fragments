#this is where i will need to import my package, need to check name
import nicolefragment
from nicolefragment import *
from nicolefragment import Molecule, MIM

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib


#This will be the file that i initalize all objects and tell it to run.  Ican do multiple runs in this file and compare at the same time!
#this file is the one that is called in the pbs.sh script

aspirin = Molecule.Molecule()
aspirin.initalize_molecule('aspirin')     
energy_MIM = MIM.do_MIM1(1, 'RHF', 'sto-3g', aspirin, 'aspirin')[0]

mol = gto.Mole()
mol.atom = #need to input string of xyz
mol.basis = sto-3g
mol.build()
opte = optimize(scf.RHF(mol).kernel())

aspirin_diff = energy_MIM - opte
print(aspirin_diff, "aspirin difference")

mol = gto.Mole()
mol.atom = #need to input string of xyz
mol.basis = sto-3g
mol.build()
opte_ben = optimize(scf.RHF(mol).kernel())

benzene = Molecule.Molecule()
benzene.initalize_molecule('benzene')
energy_MIMben = MIM.do_MIM1(1, 'RHF', 'sto-3g', benzene, 'benzene')[0]

benzene_diff = energy_MIMben - opte_ben
print(benzene_diff, "benzene difference")

#need to figure out how to do a full optimization without the fragmentation!
