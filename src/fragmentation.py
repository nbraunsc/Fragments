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
#from pyscf.geomopt import berny_solver
import psi4
import time
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util


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
        self.total = 0
        self.grad = []
        self.fullgrad = {}
        self.moleculexyz = []

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

    def total_energy(self, theory, basis):   #computes the overall energy from the fragment energies and coeffs
        for i in self.frags:
            i.build_xyz()
            i.run_pyscf(theory, basis)
            i.distribute_linkgrad()
            self.total += i.coeff*i.energy
    
    def total_gradient(self, theory, basis):    #grad_dict:individual frag grad, self.fullgrad:full molecule gradient after link atom projections, self.grad:np.array of full gradient
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
            self.grad.append(grad)

        #self.grad = np.array(self.grad)
        molecule = np.array(self.molecule.atomtable)
        self.moleculexyz = molecule[:,[1,2,3]]
        return tuple([self.total, self.grad])

    def do_geomopt(self, theory, basis):
        mol = gto.Mole()
        mol.atom = self.moleculexyz
        mol.basis = basis
        mol.build
        fake_method = as_pyscf_method(mol, self.total_gradient(theory, basis))
        new_mol = berny_solver.optimize(fake_method)
        return new_mol
       
    def opt_grad(self, theory, basis):
        opt_grad = self.do_geomopt(theory, basis)
        print(opt_grad)
    
      
   # def print_fullxyz(self):   #makes xyz input for full molecule
   #     self.molecule.atomtable = str(self.molecule.atomtable).replace('[', ' ').replace('C', '').replace('H', '').replace('O', '')
   #     self.molecule.atomtable = self.molecule.atomtable.replace('],', '\n')
   #     self.molecule.atomtable = self.molecule.atomtable.replace(',', '')
   #     self.molecule.atomtable = self.molecule.atomtable.replace("'", "")
   #     return self.molecule.atomtable
    
   # def compress_uniquefrags(self):
   #     self.compressed = []
   #     self.newcoeff = []
   #     for x in range(0, len(self.derivs)):
   #         self.compressed.append(self.derivs[x])
   #         
   #     for x in range(0, len(self.compressed)):
   #         for y in range(x+1, len(self.compressed)):
   #             if self.compressed[x] == self.compressed[y]:
   #                 self.derivs.remove(self.derivs[x])
   #                 self.derivs.remove(self.derivs[y])
   #                 self.coefflist.remove(self.coefflist[x])
   #                 self.coefflist.remove(self.coefflist[y])

            

    def do_fragmentation(self, deg, theory, basis):
        self.build_frags(deg)
        self.derivs, self.coefflist = runpie(self.unique_frag)
        self.atomlist = [None] * len(self.derivs)
        
        for i in range(0, len(self.derivs)):
            self.derivs[i] = list(self.derivs[i])

        #self.unique_derivs = self.compress_uniquefrags()
        #print(self.derivs)
        
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
        self.total_energy(theory, basis)
        self.total_gradient(theory, basis)
        self.opt_grad(theory, basis)
        #return self.total, self.grad
 
if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin')
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1, 'RHF', 'sto-3g')
    #print(frag.atomlist)
"""
    Still need to write a function to delete exact same derivatives so I am not running the same thing twice just to subtract it then adding it back.
    """
