import os
import numpy as np
import sys
from sys import argv
import xml.etree.ElementTree as ET
from runpie import *
from runpyscf import *
from Fragment import *
from Molecule import *
#from pyscf import gto, scf, hessian, mp, lib
#from pyscf.grad.rhf import GradientsBasics
#from pyscf.geomopt.berny_solver import optimize
#from berny import Berny, geomlib
#from pyscf.geomopt import berny_solver, as_pyscf_method
#from decimal import *

from scipy.optimize import minimize

class Fragmentation():
    """
    Class to do the building of molecular fragments and derivatives

    An instance of this class represents a mapping between a large molecule and a list of independent fragments with the appropriate coefficients needed to reassemble expectation values of observables. 
    """
    def __init__(self, molecule):
        self.fragment = []
        self.molecule = molecule
        self.unique_frag = []
        self.derivs = []
        self.coefflist = []
        self.atomlist = []
        self.frags = []
        self.etot = 0
        self.gradient = []
        self.fullgrad = {}  #dictonary for full molecule gradient
        self.moleculexyz = []   #full molecule xyz's

    def build_frags(self, deg):    #deg is the degree of monomers wanted
        for x in range(0, len(self.molecule.molchart)):
            for y in range(0, len(self.molecule.molchart)):
                if self.molecule.molchart[x][y] <= deg and self.molecule.molchart[x][y] != 0:
                    if x not in self.fragment:
                        self.fragment.append([x])
                        self.fragment[-1].append(y)
        for z in range(0, len(self.fragment)):
            for w in range(0, len(self.fragment)):
                if z == w:
                    continue
                if self.fragment[z][0] == self.fragment[w][0]:
                    self.fragment[z].extend(self.fragment[w][:])    #combines all prims with frag connectivity <= eta

        # Now get list of unique frags, running compress_frags function below
        self.compress_frags()
        

    def compress_frags(self): #takes full list of frags, compresses to unique frags
        frag = []
        for i in range(0, len(self.fragment)): 
            frag.append(set(self.fragment[i]))    #makes into a list of sets
        
        sign = 1
        uniquefrags = []
        for i in frag:
            add = True
            for j in frag:
                if i.issubset(j) and i != j:
                    add = False
           
            if add == True:
                if i not in uniquefrags:
                    uniquefrags.append(i)   
        self.unique_frag = uniquefrags
    
    def find_attached(self):    #finding all the attached atoms that were cut during fragmentation
        x = len(self.atomlist)
        self.attached = []
        for frag in range(0, len(self.atomlist)):
            fragi = []
            for atom in self.atomlist[frag]:
                row = self.molecule.A[atom]
                for i in range(0, len(row)):
                    if row[i] == 1.0:
                        attached = i
                        if attached not in self.atomlist[frag]:
                            fragi.append([atom, i])
                        else:
                            continue
            self.attached.append(fragi) 

    def initalize_Frag_objects(self, theory, basis):
        self.frags = []
        for fi in range(0, len(self.atomlist)):
            coeffi = self.coefflist[fi]
            attachedlist = self.attached[fi]
            self.frags.append(Fragment(theory, basis, self.atomlist[fi], self.molecule, attachedlist, coeff=coeffi))

    def global_energy(self, theory, basis):   #computes the overall energy from the fragment energies and coeffs
        for i in self.frags:
            i.build_xyz()
            i.run_pyscf(theory, basis)
            i.distribute_linkgrad()
            self.etot += i.coeff*i.energy
    
    def global_gradient(self, theory, basis):    #grad_dict:individual frag grad, self.fullgrad:full molecule gradient after link atom projections, self.gradient:np.array of full gradient
        for j in range(0, self.molecule.natoms):
            self.fullgrad[j] = np.zeros(3)

        for i in self.frags:    #projects link atom gradients back to host and supporting atoms
            for k in range(0, self.molecule.natoms):
                if k in i.grad_dict.keys():
                    self.fullgrad[k] = self.fullgrad[k] + i.coeff*i.grad_dict[k]
                else:
                    continue
        
        for l in range(0, self.molecule.natoms):    #making into numpy array
            grad = list(self.fullgrad[l])
            self.gradient.append(grad)
        self.gradient = np.array(self.gradient)
        molecule = np.array(self.molecule.atomtable)    #formatting molecule coords
        self.moleculeinput = molecule[:,[1,2,3]]
        self.moleculexyz = []
        for i in self.moleculeinput:
            x = (i[0], i[1], i[2])
            y = np.array(x)
            z = y.astype(float)
            self.moleculexyz.append(z)
        self.moleculexyz = np.array(self.moleculexyz)

    def remove_repeatingfrags(self, oldcoeff):  #this removes the derivatives that were exactly the same (i.e. this derivs would be subtracted then added right back)
        compressed = []
        coeff_position = []
        self.coefflist = []
        for i in self.derivs:
            compressed.append(i)
        for i in self.derivs:
            for x in range(0, len(compressed)):
                for y in range(0, len(compressed)):
                    if x != y:
                        if compressed[x] == compressed[y]:
                            if i == compressed[x]:
                                coeff_position.append(x)
                                self.derivs.remove(i)
        for i in range(0, len(oldcoeff)):
            if i not in coeff_position:
                self.coefflist.append(oldcoeff[i])
    
    def do_geomopt(self):
        scipy.optimize.minimize(fun, x0, args=(), method='BFGS', jac=None, tol=None, options={'gtol': 1e-08, 'maxiter': None, 'disp':True,  'eps': 1.4901161193847656e-08})

        """
        :fun -function that needs to be minimized, so in my case it is the norm of the gradient that needs to get normalized.
        :x0 -is an inital guess so it would most likely be the inital norm of the gradient or it is the inital xyz coords, energy that would be inputs to the fun function.
        :args=() -this is extra stuff that can be passed to the fun function as well as jac and hess functions
        :method -this is the type of solver, vibin mentioned the 'BFGS' method
        :jac -function that computes the gradient
        :hess -function that computes the hessian
        :tol -the tolerance for termination, float number
        :options -stuff
       """

    def do_fragmentation(self, deg, theory, basis):
        self.build_frags(deg)
        self.derivs, oldcoeff = runpie(self.unique_frag)
        self.remove_repeatingfrags(oldcoeff)  
        self.atomlist = [None] * len(self.derivs)
        
        for i in range(0, len(self.derivs)):
            self.derivs[i] = list(self.derivs[i])

        for fragi in range(0, len(self.derivs)):    #changes prims into atoms
            x = len(self.derivs[fragi])
            for primi in range(0, x):
                value = self.derivs[fragi][primi]
                atoms = list(self.molecule.prims[value])
                self.derivs[fragi][primi] = atoms
        
        for y in range(0, len(self.derivs)):
            flatlist = [ item for elem in self.derivs[y] for item in elem]
            self.atomlist[y] = flatlist     #now fragments are list of atoms
        
        for i in range(0, len(self.atomlist)):  #sorted atom numbers
            self.atomlist[i] = list(sorted(self.atomlist[i]))
        

        self.find_attached()
        self.initalize_Frag_objects(theory, basis)
        self.global_energy(theory, basis)
        self.global_gradient(theory, basis)
        return self.etot, self.gradient
 
#if __name__ == "__main__":
#    aspirin = Molecule()
#    aspirin.initalize_molecule('aspirin')
#    frag = Fragmentation(aspirin)
#    frag.do_fragmentation(1, 'MP2', 'sto-3g')


