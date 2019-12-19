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
        self.derivs = runpie(self.frag)

        """
        todo:
            sort out which atoms are attached to which atoms for doing link atom stuff.
            also, we discussed storing frags as atom lists, instead of prims.
            """

        self.frags = []
        for fi in self.derivs:
            coeff = 1
            self.frags.append(Fragment(fi,self.molecule,coeff=coeff))
            #fragi = Fragment(derivs=fi)


if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule()
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1) #argument is level of fragments wanted
    print(frag.frag, 'fraglist')
    print(frag.derivs)
    print(frag.frags)
