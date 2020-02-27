import os
import numpy as np
import sys
from sys import argv
import xml.etree.ElementTree as ET
from runpie import *
from runpyscf import *
from Fragment import *
from Molecule import *

from berny import Berny, geomlib
from pyscf.geomopt import berny_solver, as_pyscf_method
from pyscf.geomopt.berny_solver import optimize
#from scipy.optimize import minimize
from itertools import cycle

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
        self.etot_opt = 0
        self.grad_opt = []

    def build_frags(self, deg):    #deg is the degree of monomers wanted
        """
        Does the initalize fragmmentation 
        :deg - the degree of monomers wanted
        :returns the list of fragments
        """
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
        """
        Takes the full list of fragments and compresses them to only the unique fragments
        :this function is called in build_frags()
        :returns the unquie list of fragments
        """
        self.fragment = [set(i) for i in self.fragment]
        sign = 1
        uniquefrags = []
        for i in self.fragment:
            add = True
            for j in self.fragment:
                if i.issubset(j) and i != j:
                    add = False
           
            if add == True:
                if i not in uniquefrags:
                    uniquefrags.append(i)   
        self.unique_frag = uniquefrags
    
    def find_attached(self):    #finding all the attached atoms that were cut during fragmentation
        """
        Finds the atoms that were cut during the fragmentation
        :returns a list of pairs of atoms (1, 2) so atom 2 was attached to atom 1
        """
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
        """
        Initalizes the Fragment objects where link atoms get added and get fragment gets run
        """
        self.frags = []
        for fi in range(0, len(self.atomlist)):
            coeffi = self.coefflist[fi]
            attachedlist = self.attached[fi]
            self.frags.append(Fragment(theory, basis, self.atomlist[fi], self.molecule, attachedlist, coeff=coeffi))

    def remove_repeatingfrags(self, oldcoeff):
        """
        Funciton removes the repeating derivates that would be subtracted then added back again during the principle of inclusion-exculsion process
        :updates the self.derivs list and self.coefflist list that get used to intialize Fragment objects with correct attributes
        """
        compressed = []
        coeff_position = []
        self.coefflist = []
        for k in self.derivs:
            compressed.append(k)
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
    
    def energy_gradient(self, theory, basis, newcoords):
        """
        Function returns total energy (scalar) and gradient (nd.array) of molecule

        :theory is RHF or MP2
        :basis is basis set for pyscf
        :newcoords - np.array with xyz coords for the molecule, these update after each geom opt cycle
        """
        self.gradient = []  #making them back to zero for optimiztion cycles
        self.etot = 0
        for atom in range(0, len(self.molecule.atomtable)):
            x = list(newcoords[atom])
            self.molecule.atomtable[atom][1:] = x

        self.initalize_Frag_objects(theory, basis)  #reinitalizing Fragment objects with new coords
        
        for i in self.frags:
            i.build_xyz()
            i.run_pyscf(theory, basis)
            i.distribute_linkgrad()
            self.etot += i.coeff*i.energy
        
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
        return self.etot, self.gradient
    
    def write_xyz(self, name):
        """
        Function that writes an xyz file with the atom labels, number of atoms, and respective Cartesian coords
        :name - name of the Molecule class or molecule in fragmentation
        """
        molecule = np.array(self.molecule.atomtable)
        atomlabels = []
        for j in range(0, len(molecule)):
            atomlabels.append(molecule[j][0])
        coords = molecule[:,[1,2,3]]
        self.moleculexyz = []
        for i in coords:
            x = (i[0], i[1], i[2])
            y = np.array(x)
            z = y.astype(float)
            self.moleculexyz.append(z)
        self.moleculexyz = np.array(self.moleculexyz)   #formatting full molecule coords
        
        f = open("../inputs/" + name + ".xyz", "w+")
        title = ""
        f.write("%d\n%s\n" % (self.moleculexyz.size / 3, title))
        for x, atomtype in zip(self.moleculexyz.reshape(-1, 3), cycle(atomlabels)): 
            f.write("%s %.18g %.18g %.18g\n" % (atomtype, x[0], x[1], x[2]))
        f.close()

    def do_fragmentation(self, deg, theory, basis):
        """
        Funciton to fragment the molecule, make molecule xyz file for geom opt, do the principle of inculusion-exculsion
        :currently returns total energy and total gradient that was used for MIM method
        :will change this so the final geometry after geomopt will be the energy and gradient that gets used for MIM method script
        """
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
        return self.etot_opt, self.grad_opt
        #berny = Berny(molecule, steprms=0.01, stepmax=0.05, maxsteps=5)


if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin')
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1, 'RHF', 'sto-3g')
    frag.do_geomopt('aspirin', 'RHF', 'sto-3g')
    #print(frag.etot_opt)
    #print(frag.grad_opt)

