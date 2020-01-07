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
        
        self.find_attached()
        #self.compress_uniquefrags()

    def compress_uniquefrags(self): #this compresses frags based on their coefficents
        self.compressed = []
        self.newcoeff = []
        for i in range(0, len(self.atomlist)):  #sorted atom numbers
            self.atomlist[i] = list(sorted(self.atomlist[i]))
        
        for x in range(0, len(self.atomlist)):
            for y in range(0, len(self.atomlist)):
                if x != y and self.atomlist[x] == self.atomlist[y]:
                    print(self.atomlist[x], self.coefflist[x], 'x')
                    newcoeff = self.coefflist[x] + self.coefflist[y]
        
        print(self.compressed)

    def find_attached(self):
        print(self.molecule.A)
        self.attached = [None] * len(self.atomlist)
        for number in range(0, len(self.molecule.A)):
            for frag in self.atomlist:
                for atom in frag:
                    if self.molecule.A[atom][number] == 1:
                        if number not in frag:
                            print(atom, 'atom')
                            print(number, 'attached')
                            self.attached.append([atom, number])
        print(self.attached)

    def make_Frag_object(self):
        self.frags = []
        for fi in self.derivs:
            coeff = 1
            #self.frags.append(Fragment(fi,self.molecule,coeff=coeff))
            #fragi = Fragment(derivs=fi)


if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule()
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1) #argument is level of fragments wanted
    #print(frag.frag, 'fraglist')
    #print(frag.atomlist)
    #print(frag.coefflist)
    #print(frag.frags)
    
    """
        todo:
            sort out which atoms are attached to which atoms for doing link atom stuff.
            also, we discussed storing frags as atom lists, instead of prims.
            """


