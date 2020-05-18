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

    def energy_gradient(self, theory, basis, newcoords):
        f = open('../inputs/' + self.molecule + '.xyz', "r")
        h2o = psi4.geometry(f.read())
        f.close()
        
        #psi4.set_options({'basis': basis})
        energy = psi4.energy(theory + "/" + basis)
        print(energy)
        grad = psi4.gradient(theory)

    def do_geomopt(self, name, theory, basis):
        #psi4.optimize(theory + "/" + basis, molecule=h2o)
        pass
