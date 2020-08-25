import nicolefragment
from nicolefragment import *
from nicolefragment import Molecule, MIM, runpyscf, fragmentation
import time
import numpy as np

import os
import sys
from pathlib import Path

ratio_list = []
frag_list = [1, 2, 3, 4, 5]

for k in frag_list:
    diamond = Molecule.Molecule("drugs")
    diamond.initalize_molecule("diamond") #no cml extension
    frag = fragmentation.Fragmentation(diamond)
    energy_MIM = frag.do_fragmentation(k, 'RHF', 'sto-3g', "diamond")[0]
    full_energy = runpyscf.do_pyscf(diamond.atomtable, 'RHF', 'sto-3g', hess=False)[0]       #start of full molecule calc
    total_time = time.time() - start_time
    energy_error = abs(full_energy - energy_MIM)/full_energy
    ratio = total_time/energy_error
    ratio_list.append(ratio)
    
k_index = np.argmin(ratio_list)
print("Best k value for the NV center:", "", frag_list[k_index])
