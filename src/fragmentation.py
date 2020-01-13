import os
import numpy as np
import sys
from sys import argv
import xml.etree.ElementTree as ET
from tools import *
from Fragment import *
from Molecule import *

class Fragmentation():
    """
    Class to do the building of molecular fragments and derivatives

    An instance of this class represents a mapping between a large molecule and a list of independent fragments with the appropriate coefficients needed to reassemble expectation values of observables. 
    """
    def __init__(self, molecule):
        self.frag = []
        self.fragconn = []
        self.molecule = molecule
        self.derivs = []
        self.coefflist = []
        self.atomlist = []
        self.frags = []
        self.total = 0

    def build_frags(self, deg):    #deg is the degree of monomers wanted
        for x in range(0, len(self.molecule.molchart)):
            for y in range(0, len(self.molecule.molchart)):
                if self.molecule.molchart[x][y] <= deg and self.molecule.molchart[x][y] != 0:
                    if x not in self.frag:
                        self.frag.append([x])
                        self.frag[-1].append(y)
                
        for z in range(0, len(self.frag)):
            for w in range(0, len(self.frag)):
                if z == w:
                    continue
                if self.frag[z][0] == self.frag[w][0]:
                    self.frag[z].extend(self.frag[w][:])    #combines all prims with frag connectivity <= eta

        for i in range(0, len(self.frag)): 
            self.frag[i] = set(self.frag[i])    #makes into a list of sets
        
        # Now get list of unique frags, running compress_frags function below
        self.compress_frags()
        

    def compress_frags(self): #takes full list of frags, compresses to unique frags
        sign = 1
        uniquefrags = []
        for i in self.frag:
            add = True
            for j in self.frag:
                if i.issubset(j) and i != j:
                    add = False
           
            if add == True:
                if i not in uniquefrags:
                    uniquefrags.append(i)   
        self.frag = uniquefrags
    #print('done compressing frags') 
    def do_fragmentation(self, deg):
        self.build_frags(deg)
        self.derivs, self.coefflist = runpie(self.frag)
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
        self.initalize_Frag_objects()
        self.test_fragment()
        self.overall_energy()
   
   # def compress_uniquefrags(self):
   #     self.compressed = []
   #     self.newcoeff = []
   #     for x in range(0, len(self.atomlist)):
   #         for y in range(0, len(self.atomlist)):
   #             if x != y and self.atomlist[x] == self.atomlist[y]:
   #                 print(self.atomlist[x], self.coefflist[x], 'x')
   #                 newcoeff = self.coefflist[x] + self.coefflist[y]

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

    def initalize_Frag_objects(self):
        self.frags = []
        for fi in range(0, len(self.atomlist)):
            coeffi = self.coefflist[fi]
            attachedlist = self.attached[fi]
            self.frags.append(Fragment(self.atomlist[fi], self.molecule, attachedlist, coeff=coeffi))

    def test_fragment(self):
        for i in self.frags:
            i.build_xyz()
            i.run_pyscf()
        #self.frags[0].build_xyz()
        #self.frags[0].run_pyscf()
    
    def overall_energy(self):
        for i in self.frags:
            self.total += i.coeff*i.energy

    #def print_fullxyz(self):
    #    self.molecule.atomtable = str(self.molecule.atomtable).replace('[', '')
    #    self.molecule.atomtable = self.molecule.atomtable.replace(']', '\n')
    #    self.molecule.atomtable = self.molecule.atomtable.replace(',', '')
    #    print(self.molecule.atomtable)

if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule()
    #print(aspirin.atomtable)
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1) #argument is level of fragmentation wanted
    #print(frag.frags[0].grad)
    #print(frag.total)
    #print(frag.attached)
    #print(frag.atomlist)
    #print(frag.coefflist)
    #print(frag.frags) 

"""
    Still need to write a function to delete exact same derivatives so I am not running the same thing twice just to subtract it then adding it back.
    """
