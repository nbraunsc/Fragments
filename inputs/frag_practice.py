import nicolefragment
from nicolefragment import *
from nicolefragment import Molecule, MIM, runpyscf, fragmentation
import time
import numpy as np

import os
import sys
from pathlib import Path

frag_list = [1, 2, 3, 4, 5]
best_list = []
molecule_classlist = ["drugs"]
filenames = []


#directory = os.fsencode(str(/home/nbraunsc/Projects/Fragments/inputs/))
#os.path.abspath(os.curdir)

current_dir = os.getcwd()

#for i in molecule_classlist:
#directory = os.chdir(i)
directory = current_dir + "/drugs/"
x = 0
for file_name in os.listdir(directory):
    ratio_list = []
    if file_name.endswith(".cml"):
        filenames.append(file_name)
        for k in frag_list:
            start_time = time.time()
            filename_nocml = Path(file_name).stem
            z = str(Path(file_name).stem)
            a_k = Molecule.Molecule("drugs")
            a_k.initalize_molecule(z) #no cml extension
            frag = fragmentation.Fragmentation(a_k)
            energy_MIM = frag.do_fragmentation(k, 'RHF', 'sto-3g', z)[0]
            full_energy = runpyscf.do_pyscf(a_k.atomtable, 'RHF', 'sto-3g', hess=False)[0]       #start of full molecule calc
            total_time = time.time() - start_time
            energy_error = abs(full_energy - energy_MIM)/full_energy
            ratio = total_time/energy_error
            ratio_list.append(ratio)
        k_index = np.argmin(ratio_list)
        best_list.append(frag_list[k_index])
    

for i in range(0, len(filenames)):
    print(filenames[i], "", "Best k value:", "", best_list[i])

#x = np.argmin(ratio_list)      #smaller time to error = best fragmentation level
f = open("molecule_class", "a")
f.write("Drugs\n")
f.write("Best k value =")
f.write(x)
f.close()


