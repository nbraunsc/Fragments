import os
import numpy as np
import sys
from sys import argv
import xml.etree.ElementTree as ET
from .cov_rad import *

class Molecule():
    """
    Class to sort through cml input of molecule, compute attributes that will not change for molecule
    :makes into xyz format
    :builds primatives
    :builds connectivity charts of atom-atom and prim-prim
    :parse_cml(filename), file path needs to be specified
    """

    def __init__(self, mol_class=str()):
        #number of atoms
        self.natoms = {} 
        #atom coords with index and element, type:list
        self.atomtable = []
        #table with labels of integers for atoms, starting at 0, type: np array
        self.bond_lables = []
        #table with just bond orders, type: np array
        self.bond_order = []
        #table with labels and bond order, type:numpy.array
        self.bond_table = []
        #atom connectivity matrix
        self.A = []
        #list of primiatives
        self.prims = []
        #prim connect chart
        self.primchart = []
        #higher order prim conn chart
        self.molchart = []
        self.covrad = form_covalent_radii()
        self.optxyz = []
        self.mol_class = mol_class
        
    def initalize_molecule(self, file_name):
        current_dir = os.getcwd()
        #x = current_dir + "/" + self.mol_class + "/" + file_name + ".cml"
        #x = "../inputs/" + self.mol_class + "/" + file_name + ".cml"
        directory = "/home/nbraunsc/Projects/Fragments/inputs/drugs/" + file_name + ".cml"
        self.parse_cml(directory)
        self.build_molmatrix(2)
        #os.chdir("../")

    def parse_cml(self, filename):
        """
        Takes the cml file and converts it into xyz coords still keeping the bonding information
        :filename - name of the file name for the molecule wanted to run
        """
        #self.filename = filename
        tree = ET.parse(filename)
        #self.tree = tree
        root = tree.getroot()
        molecule = root
        #self.molecule = molecule
        atomArray = root[3]    #root[0]
        self.atomArray = atomArray
        bondArray = root[4]     #root[1]
        #self.bondArray = bondArray        
        self.natoms = len(self.atomArray)
        self.A = np.zeros( (self.natoms,self.natoms)) 
        self.atomtable = [[0 for x in range(4)] for y in range(self.natoms)]
        for atomi in range(0, self.natoms):
            self.atomtable[atomi][0] = atomArray[atomi].attrib['elementType']
            self.atomtable[atomi][1] = float(atomArray[atomi].attrib['x3'])
            self.atomtable[atomi][2] = float(atomArray[atomi].attrib['y3'])
            self.atomtable[atomi][3] = float(atomArray[atomi].attrib['z3'])
        #start extracting bond order
        for bondi in bondArray:
            a12 = bondi.attrib['atomRefs2'].split()
            #assert is making all first part of indexes = a
            assert(a12[0][0] == "a")
            assert(a12[1][0] == "a")
            #takes a1 and makes it a0 so index of element is same in bondorder
            a1 = np.array([int(a12[0][1:])-1])
            a2 = np.array([int(a12[1][1:])-1])
            self.bond_labels = np.append(a1,a2)
            self.bond_order = np.array([int(bondi.attrib["order"])])
            self.bond_table = np.append(self.bond_labels, self.bond_order)
            #these need to be within for loop, don't know if need self.
            x = self.bond_table[0]
            y = self.bond_table[1]
            z = self.bond_table[2]
            self.A[x][y] = z
            self.A[y][x] = z
        self.build_prims()
        self.build_primchart()
    
    def build_prims(self):
        """
        Builds the primiative fragments (smallest possible fragments)
        :these primiatitves are made by not cutting hydrogen bonds or higher than single bonds
        :funciton gets called in parse_cml()
        :returns a list of primiataives
        """
        for i in range(0, len(self.atomtable)):
            if self.atomtable[i][0] != "H":
                self.prims.append([i])
                for j in range(0, len(self.atomtable)):
                    if self.A[i][j] == 1 and self.atomtable[j][0] == "H" or self.A[i][j] > 1: #anything bonded to an H or non-single bond pull into prim
                        self.prims[-1].append(j)
                    for k in range(0, len(self.atomtable)):
                        if self.A[i][j] > 1 and self.A[j][k] == 1 and self.atomtable[k][0] == "H": #check if atom pulled in is bonded to H or higher order bond, pull it into prim also
                            self.prims[-1].append(j)
                            self.prims[-1].append(j)
                            self.prims[-1].append(k)
        
        #deletes duplicates within a prim
        self.prims = list(set(x) for x in self.prims)
        for i in range(0, len(self.prims)):
            self.prims[i] = tuple(sorted(self.prims[i]))
        self.prims = set(self.prims)
    
    def build_primchart(self):
        """
        Builds a primiative array (nd array) where row and column are associated with each primative
        :returns a matrix called A where if the primiatives are connected it adds a 1 in that spot
        """
        self.prims = list(self.prims)
        self.primsleng = len(self.prims)
        self.primchart = np.zeros( (self.primsleng,self.primsleng))
        for prim1 in range(0, len(self.prims)):
            for atomi in self.prims[prim1]:
                for prim2 in range(0, len(self.prims)):
                    for atomj in self.prims[prim2]:
                        if prim1 == prim2:
                            continue
                        if self.A[atomi][atomj] != 0:
                            self.primchart[prim1][prim2] = 1
                            self.primchart[prim2][prim1] = 1
    
    def build_molmatrix(self, i):
        """
        Builds the whole molecule matrix (nd array) where it shows what order each prim is connected: to eachtother
        :built through a series of matrix manipulations (need to review this)
        :i must be 2 to start the recursive funciton correctly
        :returns nd array that is used for fragment building
        """
        eta = self.natoms
        if i == 2:
            self.molchart = self.primchart.dot(self.primchart)
            np.fill_diagonal(self.molchart, 0)
            for x in range(0, len(self.prims)):
                for y in range(0, len(self.prims)):
                    if self.molchart[x][y] != 0:
                        self.molchart[x][y] = i
            
            for x in range(0, len(self.prims)):      #checkpoint
                for y in range(0, len(self.prims)):
                    if self.primchart[x][y] != 0:
                        self.molchart[x][y] = 0
            self.molchart = np.add(self.molchart, self.primchart)
        self.molchartnew = []   
        if i > 2:
            self.molchartnew = self.molchart.dot(self.primchart)
            np.fill_diagonal(self.molchartnew, 0)
            for x in range(0, len(self.prims)):
                for y in range(0, len(self.prims)):
                    if self.molchartnew[x][y] != 0:
                        self.molchartnew[x][y] = i
            
            for x in range(0, len(self.prims)):     #checkpoint 
                for y in range(0, len(self.prims)):
                    if self.molchart[x][y] != 0:
                        self.molchartnew[x][y] = 0
            self.molchart = np.add(self.molchart, self.molchartnew)

        if i < eta:     #recursive part of function
            i = i+1
            self.build_molmatrix(i)

if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin')
