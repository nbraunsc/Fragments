try:
    import os, sys
    sys.path.insert(1, os.path.abspath('/scratch/psilocaluser/conda-builds/psi4-docs-multiout_1589298426985/work/build/stage//scratch/psilocaluser/conda-builds/psi4-docs-multiout_1589298426985/_h_env_placehold_placehold_place/lib/python3.7/site-packages'))
except ImportError:
    pass

import psi4
import os
import numpy as np
from runpie import *
from runpyscf import *
from Fragment import *
from Molecule import *

from berny import Berny, geomlib
from pyscf.geomopt import berny_solver, as_pyscf_method
from pyscf.geomopt.berny_solver import optimize
from itertools import cycle

class Psi4():
    """
    Virtual class to run psi4 
    """

    def __init__(self, molecule):
        self.molecule = molecule

    def energy_gradient(self):
    #def energy_gradient(self, theory, basis, newcoords):
        f = open('../inputs/carbonylavo.xyz', "r")
        h2o = psi4.geometry(f.read())
        f.close()
        
        #psi4.set_options({'basis': basis})
        psi4.energy(theory + "/" + basis)
        grad = psi4.gradient(theory)

    def do_geomopt(self, name, theory, basis):
        #psi4.optimize(theory + "/" + basis, molecule=h2o)
        pass
