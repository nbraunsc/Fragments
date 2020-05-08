import nicolefragment
from nicolefragment import *
from nicolefragment import Molecule, MIM, runpyscf
import time
import numpy as np

import os
import sys
from pathlib import Path

frag_list = [1, 2, 3, 4, 5]
ratio_list = []
molecule_classlist = ["alcohols", "amines"]


#directory = os.fsencode(str(/home/nbraunsc/Projects/Fragments/inputs/))
#os.path.abspath(os.curdir)

current_dir = os.getcwd()

for i in molecule_classlist:
    #directory = os.chdir(i)
    directory = current_dir + "/" + i + "/"
    x = 0
    for file_name in os.listdir(directory):
        if file_name.endswith(".cml"):
            for k in frag_list:
                start_time = time.time()
                filename_nocml = Path(file_name).stem
                z = str(Path(file_name).stem)
                a_k = Molecule.Molecule(i)
                print(a_k)
                a_k.initalize_molecule(z) #no cml extension
                print(a_k.atomtable)
                energy_MIM = MIM.do_MIM1(k, 'RHF', 'sto-3g', a_k, z)[0]
                full_energy = runpyscf.do_pyscf(a_k.atomtable, 'RHF', 'sto-3g', hess=False)[0]       #start of full molecule calc
                total_time = time.time() - start_time
                energy_error = abs(full_energy - energy_MIM)/full_energy
                ratio = total_time/energy_error
                ratio_list.append(ratio)
            x = np.argmin(ratio_list)      #smaller time to error = best fragmentation level
        else:
            continue
    
    os.chdir("../")     #writing results for each molecule class
    f = open("molecule_class", "a")
    f.write(i)
    f.write("Best k value =")
    f.write(x)
    f.close()


