#this is where i will need to import my package, need to check name
from MIM_fragment import *


#This will be the file that i initalize all objects and tell it to run.  Ican do multiple runs in this file and compare at the same time!
#this file is the one that is called in the pbs.sh script

aspirin = Molecule()
aspirin.initalize_molecule('aspirin')     
do_MIM1(1, 'RHF', 'sto-3g', aspirin, 'aspirin')        
