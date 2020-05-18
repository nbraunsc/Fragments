import os
import numpy as np
import sys
from sys import argv
from runpie import *
from runpyscf import *
from Fragment import *
from Molecule import *

from berny import Berny, geomlib
from pyscf.geomopt import berny_solver, as_pyscf_method
from pyscf.geomopt.berny_solver import optimize
from itertools import cycle


class Pyscf():
    """
    Virtual class to run pyscf in the energy_gradient() and geom_opt()
    """
    def __init__(self, molecule):
        self.molecule = molecule


    def energy_gradient(self, theory, basis, newcoords):
        """
        Function returns total energy (scalar) and gradient (nd.array) of molecule

        :theory is RHF or MP2
        :basis is basis set for pyscf
        :newcoords - np.array with xyz coords for the molecule, these update after each geom opt cycle
        """
        self.gradient = []  #making them back to zero for optimiztion cycles
        self.etot = 0
        for atom in range(0, len(self.molecule.atomtable)): #updates coords in Molecule class
            x = list(newcoords[atom])
            self.molecule.atomtable[atom][1:] = x

        self.initalize_Frag_objects(theory, basis)  #reinitalizing Fragment objects with new coords
        self.gradient = np.zeros((self.molecule.natoms, 3)) 
        for i in self.frags:
            i.run_pyscf(theory, basis)
            self.etot += i.coeff*i.energy
            self.gradient += i.coeff*i.grad
        return self.etot, self.gradient
   
    def do_geomopt(self, name, theory, basis):
        """
        Completes the geometry optimization using pyberny from pyscf
        :name - name of Molecule() object to help with file path
        :theory - theory for pyscf either RHF or MP2
        :basis - basis set for pyscf
        :returns the optimized geometry of the full molecule
        :I will eventually set it up where you can change different parameters for geomopt
        """
        self.write_xyz(name)
        os.path.abspath(os.curdir)
        #os.chdir('../inputs/' + self.molecule.mol_class)
        os.chdir('../inputs/')
        optimizer = Berny(geomlib.readfile(os.path.abspath(os.curdir) + '/' + name + '.xyz'), debug=True)
        for geom in optimizer:
            solver = self.energy_gradient(theory, basis, geom.coords)
            optimizer.send(solver)
            self.etot_opt = solver[0]
            self.grad_opt = solver[1]
            print("\n", "Energy = ", solver[0], "\n")
            print("Gradients:", "\n", solver[1], "\n")
        relaxed = geom
        print("Converged geometry coords:", "\n", relaxed.coords)
        self.molecule.optxyz = relaxed.coords
        os.chdir('../')
        return self.etot_opt, self.grad_opt
        #berny = Berny(molecule, steprms=0.01, stepmax=0.05, maxsteps=5)
