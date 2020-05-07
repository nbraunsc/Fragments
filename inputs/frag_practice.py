import nicolefragment
from nicolefragment import *
from nicolefragment import Molecule, MIM, runpyscf
import time
import numpy as np

from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib

frag_list = [1, 2, 3, 4, 5]
ratio_list = []

for k in frag_list:
    start_time = time.time()
    largermol = Molecule.Molecule()
    largermol.initalize_molecule('largermol')
    energy_MIM = MIM.do_MIM1(k, 'RHF', 'sto-3g', largermol, 'largermol')[0]
    
    full_energy = runpyscf.do_pyscf(largermol.atomtable, 'RHF', 'sto-3g', hess=False)[0]
    total_time = time.time() - start_time
    energy_error = abs(full_energy - energy_MIM)/full_energy
    ratio = total_time/energy_error
    ratio_list.append(ratio)

x = np.argmin(ratio_list)       #smaller time to error = best fragmentation level
print("Best k value for clustering:", x)
