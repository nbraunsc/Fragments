import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np

np.set_printoptions(suppress=True, precision=5)
    
largermol = Molecule.Molecule()
largermol.initalize_molecule('largermol')
        
"""do_MIM1(deg, frag_type,  theory, basis, Molecule, opt=False, step=0.001)"""
do_MIM1(1.8, 'distance', 'RHF', 'sto-3g', largermol, opt=False, step=0.001)        #uncomment to run MIM1
    
"""do_MIM2(frag_type, frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule, opt=False)"""
#do_MIM2('distance', 1.0, 'MP2', 'sto-3g', 1.8, 'RHF', 'sto-3g', largermol, opt=False) #uncomment to run MIM2
