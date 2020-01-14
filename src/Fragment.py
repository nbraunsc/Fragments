from linkatoms import *
from runpyscf import *
import string

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

    def build_xyz(self):    #builds input with atom label, xyz coords, and link atoms as a string
        for atom in self.prims:
            atom_xyz = str(self.molecule.atomtable[atom]).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
            self.inputxyz += atom_xyz
        for pair in self.attached:
            linkatom_xyz = str(add_linkatoms(pair[0], pair[1], self.molecule)).replace('[', '')
            linkatom_xyz = str(add_linkatoms(pair[0], pair[1], self.molecule)).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
            self.inputxyz += linkatom_xyz
    
    def run_pyscf(self, theory, basis):
        #print(self.prims)
        #print(self.inputxyz, '\n')
        self.energy, self.grad = do_pyscf(self.inputxyz, self.theory, self.basis) #'sto-3g')
        
        x = len(self.prims)
        for i in range(x, len(self.grad)):
            mag = np.linalg.norm(self.grad[i])
