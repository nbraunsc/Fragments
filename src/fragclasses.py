import os
import numpy as np
import sys
from sys import argv
import xml.etree.ElementTree as ET
#from cov_rad.py import *

class Molecule():
    def __init__(self):
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

    def parse_cml(self, filename):
        self.filename = filename
        tree = ET.parse(filename)
        self.tree = tree
        root = tree.getroot()
        molecule = root
        self.molecule = molecule
        atomArray = root[0]
        self.atomArray = atomArray
        bondArray = root[1]
        self.bondArray = bondArray        
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
            
    def build_prims(self):
        for i in range(0, len(self.atomtable)):
            if self.atomtable[i][0] != "H":
                self.prims.append([i])
                for j in range(0, len(self.atomtable)):
                    if self.A[i][j] == 1 and self.atomtable[j][0] == "H" or self.A[i][j] > 1: #anything bonded to an H or non-single bond pull into prim
                        self.prims[-1].append(j)
                    for k in range(0, len(self.atomtable)):
                        if self.A[i][j] > 1 and self.A[j][k] == 1 and self.atomtable[k][0] == "H": #check if atom pulled in is bonded to H or higher order bond, pull it into prim also
                            self.prims[-1].append(j)
                            self.prims[-1].append(k)
        
        #deletes duplicates within a prim
        self.prims = list(set(x) for x in self.prims)
        for i in range(0, len(self.prims)):
            self.prims[i] = tuple(sorted(self.prims[i]))
        self.prims = set(self.prims)
    
    #if spot in A is non zero add 1 to primchart in row and column of prim1 and prim2 
    def build_primchart(self):
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
    
    def build_molmatrix(self, eta, i):    #i must 2 to start
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
            self.build_molmatrix(eta, i)
    
class Fragmentation():
    def __init__(self, molecule):
        self.frag = []
        #unqiue fragments
        self.uniquefrags = []
        #layer two of frags
        self.fragconn = []
        self.molecule = molecule

    def build_frags(self, deg):    #deg is the degree of fragments wanted
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
                    self.frag[z].extend(self.frag[w][:])    #combines all prims with frag connectivity <= eta, list of lists

        for i in range(0, len(self.frag)): 
            self.frag[i] = set(self.frag[i])    #makes into a list of sets
    
    def compress_frags(self): #takes full list of frags, compresses to unique frags
        sign = 1
        for i in self.frag:
            add = True
            for j in self.frag:
                if i.issubset(j) and i != j:
                    add = False
           
            if add == True:
                if i not in self.uniquefrags:
                    self.uniquefrags.append(i)   
        print(self.uniquefrags)
    
    def build_frags2(self):  #builds second layer of frags/frag intersections
        sign = -1
        for fi in range(0, len(self.uniquefrags)):
            for fj in range(0, len(self.uniquefrags)):
                if fi < fj:
                    inter = self.uniquefrags[fi].intersection(self.uniquefrags[fj])
                    self.fragconn.append(inter)
        print(self.fragconn)

class Fragment():
    def __init__(self, molecule, fragmentation):
        self.molecule = molecule
        self.fragmentation = fragmentation


    def add_links(self):
        linkatoms = []
        for frag in range(0, len(self.uniquefrags)):
            for prim in self.uniquefrags[frag]:
                for prim2 in range(0, len(self.prims)):
                    if self.molecule.molchart[prim][prim2] == 1 and prim2 not in self.uniquefrags[frag]:
                        for i in self.molecule.prims[prim]:
                            for j in self.molecule.prims[prim2]:
                                if self.molecule.A[i][j] == 1:
                                    linkatoms.append([i])
                                    linkatoms[-1].append(j)
       
        for i in range(0, len(linkatoms)):
            linkatoms[i] = sorted(set(linkatoms[i]))
            linkatoms[i] = set(linkatoms[i])
        
        for i in linkatoms:
            add = True
            for j in linkatoms:
                if i.issubset(j) and i != j:
                    add = False
            if add == True:
                if i not in self.linkatoms:
                    self.linkatoms.append(i)

#       for pair in self.linkatoms:
#            for atom in pair:
#                if 
#                       
        
# find norm and vector between those connecting atoms
# of atom in prim, find cov_rad, do the ratio thing between a normal c-h bond and cov_rad, add H link along that vector at that distance ratio
# do this for fragments and fragment intersestions



if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.parse_cml("../inputs/aspirin.cml")
    aspirin.build_prims()
    aspirin.build_primchart()
    aspirin.build_molmatrix(10, 2)

    frag = Fragmentation(aspirin)
    frag.build_frags(1)
    frag.compress_frags()
    frag.build_frags2()
