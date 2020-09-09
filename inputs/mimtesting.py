import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import ray
import os
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib

np.set_printoptions(suppress=True, precision=5)
largermol = Molecule.Molecule()
largermol.initalize_molecule('largermol')
do_MIM1(1.8, 'distance', 'RHF', 'sto-3g', largermol, opt=False, step_size=0.001)        #uncomment to run MIM1
