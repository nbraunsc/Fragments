import nicolefragment
from nicolefragment import *
from nicolefragment import Molecule, MIM, runpyscf
import time
import numpy as np

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib

import os
import sys
from pathlib import Path

frag_list = [1, 2, 3, 4, 5]
ratio_list = []
molecule_classlist = ["alcohols", "amines"]


#directory = os.fsencode(str(/home/nbraunsc/Projects/Fragments/inputs/))
#os.path.abspath(os.curdir)

current_dir = os.getcwd()
ratio_list = []

for i in molecule_classlist:
    #directory = os.chdir(i)
    directory = current_dir + "/" + i + "/"
    ratio_list.clear()
    for file_name in os.listdir(directory):
        if file_name.endswith(".cml"):
            for k in frag_list:
                print(i, "class", file_name, "molecule name")
                start_time = time.time()
                filename_nocml = Path(file_name).stem
                filename_nocml = Molecule.Molecule(i)
                print(filename_nocml, "molecule name with no cml")
                filename_nocml.initalize_molecule(Path(file_name).stem) #no cml extension
                print(filename_nocml.atomtable)
                energy_MIM = MIM.do_MIM1(k, 'RHF', 'sto-3g', filename_nocml, Path(file_name).stem)[0]
                full_energy = runpyscf.do_pyscf(filename_nocml.atomtable, 'RHF', 'sto-3g', hess=False)[0]       #start of full molecule calc
                total_time = time.time() - start_time
                energy_error = abs(full_energy - energy_MIM)/full_energy
                ratio = total_time/energy_error
                print(ratio, "this is the ratio")
                ratio_list.append(ratio)
    x = np.argmin(ratio_list)      #smaller time to error = best fragmentation level
    os.chdir("../")     #writing results for each molecule class
    f = open("molecule_class", "a")
    f.write(i)
    f.write("Best k value =")
    f.write(x)
    f.close()
    
# need to figure out how to say if fragmentation yields the full molecule, void that entry and 
# end for loop for the k numbers

