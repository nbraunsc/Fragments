#this is where i will need to import my package, need to check name
import nicolefragment
from nicolefragment import *
from nicolefragment import Molecule, MIM


#This will be the file that i initalize all objects and tell it to run.  Ican do multiple runs in this file and compare at the same time!
#this file is the one that is called in the pbs.sh script

aspirin = Molecule.Molecule()
aspirin.initalize_molecule('aspirin')     
energy_MIM = MIM.do_MIM1(1, 'RHF', 'sto-3g', aspirin, 'aspirin')[0]

aspirin1 = Molecule.Molecule()
aspirin1.initalize_molecule('aspirin')     
energy_full = MIM.do_MIM1(1, 'full', 'sto-3g', aspirin1, 'aspirin')[0]

aspirin_diff = energy_MIM - energy_full
print(aspirin_diff, "aspirin difference")

benzene = Molecule.Molecule()
benzene.initalize_molecule('benzene')
energy_MIMben = MIM.do_MIM1(1, 'RHF', 'sto-3g', benzene, 'benzene')[0]

benzene1 = Molecule.Molecule()
benzene1.initalize_molecule('benzene')
energy_fullben = MIM.do_MIM1(1, 'full', 'sto-3g', benzene1, 'benzene')[0]

benzene_diff = energy_MIMben - energy_fullben
print(benzene_diff, "benzene difference")
