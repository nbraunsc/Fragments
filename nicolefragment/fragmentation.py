import os
import numpy as np
import sys
from sys import argv
import xml.etree.ElementTree as ET
from runpie import *
from runpyscf import *
from runpsi4 import *
from Fragment import *
from Molecule import *
from itertools import cycle

class Fragmentation():
    """
    Fragmentation is a child class of Molecule().
    
    Used to build the molecular fragments and their derivatives.
    
    An instance of this class represents a mapping between a large molecule and a list of independent fragments with the appropriate coefficients needed to reassemble expectation values of observables. 
    """
    
    def __init__(self, molecule, chem_software):
        self.fragment = []
        self.molecule = molecule
        self.unique_frag = []
        self.derivs = []
        self.coefflist = []
        self.atomlist = []
        self.frags = []
        self.etot = 0
        self.gradient = []
        self.hessian = []
        self.fullgrad = {}  #dictonary for full molecule gradient
        self.fullhess = {}
        self.moleculexyz = []   #full molecule xyz's
        self.etot_opt = 0
        self.grad_opt = []
        self.chem_software = chem_software

    def build_frags(self, deg):    #deg is the degree of monomers wanted
        """ Performs the initalize fragmmentation 
        
        This will fragment the full molecule at the level of deg specified.
        Also runs self.compress_frags() to delete repeating fragments.
        
        Parameters
        ----------
        deg : int
            The degree of fragmentation wanted. Larger deg results in larger fragments.
        
        Returns
        -------
        self.fragment : list
            List of fragments containing index of primiatives that are within deg
        
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
        """ Takes the full list of fragments and compresses them to only the unique fragments.
        
        This functions compresses self.fragment into self.uniquefrag. 
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.unique_frag : list
            List of sets containing only the unqie fragments 
        
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
        """ Finds the atoms that were cut during the fragmentation.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.attached : list of lists
            List of pairs of atoms (a, b) len of # of fragments, where a is in fragment and atom b was the 
            atom that was cut.
        
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
        """ Initalizes the Fragment() objects
        
        Fragment is another child class of Fragmentaion() where link atoms get added, 
        fragment energy, gradient and hessian is calcuated.
        
        Parameters
        ----------
        theory : str
            Level of theory each fragment should be run at
        basis : str
            Basis set wanted for calculation
        
        Returns
        -------
        none
        
        """

        self.frags = []
        for fi in range(0, len(self.atomlist)):
            coeffi = self.coefflist[fi]
            attachedlist = self.attached[fi]
            self.frags.append(Fragment(theory, basis, self.atomlist[fi], self.molecule, attachedlist, coeff=coeffi))

    def remove_repeatingfrags(self, oldcoeff):
        """ Removes the repeating derivates. Function gets called in self.do_fragmentation().
        
        The repeating derivates are the exact same derivates that would be added then subtracted
        again during the principle of inclusion-exculsion process of MIM.  This also updates teh self.derivs
        and self.coefflist that are using with Fragment() class instances.
        
        Parameters
        ----------
        oldcoeff : list
            List of coefficients that is the output of runpie() function that is called in self.do_fragmentation().
        
        Returns
        -------
        self.coefflist : list
            List of coefficents where index of coeff correlates to that fragment
            
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
    
    def energy_gradient(self, theory, basis, newcoords, name):
        """ Function returns total energy and gradient of global molecule.
        
        This function holds virtual functions for different chemical software.  A software
        must be implemented inorder to run this function.  
        
        Parameters
        ----------
        theory : str
            Theory of calculation wanted
        basis : str
            Basis set wanted for calculations
        newcoords : npdarray
            Contains xyz coords for the full molecule. These get updated after each geometry optimization cycle.
        
        Returns
        -------
        self.etot : float
            Energy of the full molecule
        self.gradient : ndarray
            Gradient of the full molecule
        
        """

        self.gradient = np.zeros((self.molecule.natoms, 3)) #setting them back to zero
        self.etot = 0

        for atom in range(0, len(self.molecule.atomtable)): #makes newcoords = self.molecule.atomtable
            x = list(newcoords[atom])
            self.molecule.atomtable[atom][1:] = x

        self.initalize_Frag_objects(theory, basis)  #reinitalizing Fragment objects with new coords
        
        for i in self.frags:
            if self.chem_software == 'pyscf':
                i.run_pyscf(theory, basis)
                self.etot += i.coeff*i.energy
                self.gradient += i.coeff*i.grad
                #return self.etot, self.gradient
                #self.etot, self.gradient = py.energy_gradient(theory, basis, newcoords)
            if self.chem_software == 'psi4':
                i.run_psi4(theory, basis, name)
                self.etot += i.coeff*i.energy
                #self.gradient += i.coeff*i.grad
                #self.etot, self.gradient = Psi4.energy_gradient(theory, basis, newcoords)
                #return self.etot, self.gradient
            else:
                raise Exception("NoChemicalSoftwareImplemented_energy_gradient")
        return self.etot#, self.gradient

    def write_xyz(self, name):
        """ Writes an xyz file with the atom labels, number of atoms, and respective Cartesian coords

        Parameters
        ----------
        name : str
            Name of the molecule, must be same name as input.cml file
        
        Returns
        -------
        none
        
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
        """ Main executeable for Fragmentation () class

        This function fragments the molecule, runs principle of inclusion-exculsion,
        removes repeating fragments, creates attached pairs list, initalizes instances of Fragment() class.
        
        Parameters
        ----------
        deg : int
            Degree of fragmentation wanted, gets used in self.build_frags()
        theory : str
            Level of theory wanted to be used in calculation
        basis : str
            Basis set name for calculation
        
        Returns
        -------
        none
        
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
        
    def do_geomopt(self, name, theory, basis):
        """ Completes the geometry optimization using pyberny from pyscf.
        
        Parameters
        ----------
        name : str
            Name of Molecule() object to help with file path
        theory : str
            Level of theory for calculation
        basis : str
            Basis set name for calculation
        
        Returns
        -------
        self.etot_opt : float
            Optimized energy for full molecule
        self.grad_opt : ndarray
            Optimized gradient for full molecule
        
        """
        self.write_xyz(name)
        os.path.abspath(os.curdir)
        #os.chdir('../inputs/' + self.molecule.mol_class)
        os.chdir('../inputs/')
        optimizer = Berny(geomlib.readfile(os.path.abspath(os.curdir) + '/' + name + '.xyz'), debug=True)
        for geom in optimizer:
            solver = self.energy_gradient(theory, basis, geom.coords, name)
            optimizer.send(solver)
            self.etot_opt = solver[0]
            #self.grad_opt = solver[1]
        relaxed = geom
        print("\n", "##########################", '\n', "#       Converged!       #", '\n', "##########################") 
        print('\n', "Energy = ", self.etot_opt)
        #print('\n', "Converged_Gradient:", "\n", self.grad_opt)
        self.molecule.optxyz = relaxed.coords
        os.chdir('../')
        return self.etot_opt#, self.grad_opt
       
    def compute_Hessian(self, theory, basis):
        """
        Computes the overall Hessian for the molecule after the geometry optimization is completed.
        :Also does link atom projects for the Hessians
        """
        self.hessian = np.zeros((self.molecule.natoms, self.molecule.natoms, 3, 3)) 
        for i in self.frags:
            i.do_Hessian()
            self.hessian += i.hess*i.coeff

        #x = np.reshape(self.hessian, (self.hessian.shape[0]*3, self.hessian.shape[1]*3))
        #hess_values, hess_vectors = LA.eigh(x)
        #print(hess_vectors, "eigenvectors aka directions of normal modes")
        #print(hess_values, "eigenvalues, aka frequencies"
        return self.hessian, hess_values, hess_vectors

############################################################# 
# IDEAS TO MAKE OWN HESSIANS:                               #
# - use psi4 but then i would have to convert everything to #
#   psi4 format, these are numerical Hessians anyway        #
# - use autograd like in the example below, these pretty    #
#   much just computes the jacobian which is the same thing #
#   as the hessian (need to check the theory on this)       #
# - also need to check if my mp2 gradients are actual mp2   #
#   gradients, vibin thinks pyscf is just giving hf grads   #
# - autograd will also be able to give me the analytical    #
#   gradients and hessians for higher levels of theory      #
# - i also need to do finite difference to check hessians   #
#############################################################
"""
from autograd import elementwise_grad as egrad
from autograd import jacobian
import autograd.numpy as np

def func(x):
    return np.sin(x[0]) * np.sin(x[1])

x_value = np.array([0.0, 0.0])  # note inputs have to be floats
H_f = jacobian(egrad(func))  # returns a function
print(H_f(x_value))
"""

if __name__ == "__main__":
    carbonylavo = Molecule()
    carbonylavo.initalize_molecule('carbonylavo')
    frag = Fragmentation(carbonylavo, 'psi4')
    frag.do_fragmentation(1, 'MP2', 'sto-3g')
    frag.do_geomopt('carbonylavo', 'MP2', 'sto-3g')
    #frag.compute_Hessian('MP2', 'sto-3g')

