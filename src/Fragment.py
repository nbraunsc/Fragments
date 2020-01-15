from runpyscf import *
import string
import numpy as np

class Fragment():
    """
    Class to store a list of primitives corresponding to a molecular fragment
    """
    def __init__(self, theory, basis, prims, molecule, attached=[], coeff=1):
        self.prims = prims 
        self.molecule = molecule 
        self.coeff = coeff 
        self.attached = attached
        self.inputxyz = str()
        self.energy = 1
        self.grad = []
        self.theory = theory
        self.basis = basis

    def __str__(self):
        out = "Frag:"
        out += str(self.prims)
        out += str(self.coeff)
        out += str(self.attached)
        return out 

    def __repr__(self):
        return str(self)

    def add_linkatoms(self, atom1, attached_atom, molecule):
        atom1_element = molecule.atomtable[atom1][0]
        attached_atom_element = molecule.atomtable[attached_atom][0]
        cov_atom1 = molecule.covrad[atom1_element][0]
        cov_attached_atom = molecule.covrad[attached_atom_element][0]
        atom_xyz = np.array(molecule.atomtable[atom1][1:])
        attached_atom_xyz = np.array(molecule.atomtable[attached_atom][1:])
        vector = attached_atom_xyz - atom_xyz
        dist = np.linalg.norm(vector)
        h = 0.32
        factor = (h + cov_atom1)/(cov_atom1 + cov_attached_atom)
        new_xyz = list(factor*vector+atom_xyz)
        new_xyz.insert(0, 'H')
        return new_xyz
    
    def build_xyz(self):    #builds input with atom label, xyz coords, and link atoms as a string
        for atom in self.prims:
            atom_xyz = str(self.molecule.atomtable[atom]).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
            self.inputxyz += atom_xyz
        for pair in self.attached:
            linkatom_xyz = str(self.add_linkatoms(pair[0], pair[1], self.molecule)).replace('[', '')
            linkatom_xyz = str(self.add_linkatoms(pair[0], pair[1], self.molecule)).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
            self.inputxyz += linkatom_xyz
    
    def run_pyscf(self, theory, basis):
        #print(self.prims)
        #print(self.inputxyz, '\n')
        self.energy, self.grad = do_pyscf(self.inputxyz, self.theory, self.basis) #'sto-3g')
        
        x = len(self.prims)
        for i in range(x, len(self.grad)):
            mag = np.linalg.norm(self.grad[i])
