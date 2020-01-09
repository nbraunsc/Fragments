from linkatoms import *
from runpyscf import *

class Fragment():
    """
    Class to store a list of primitives corresponding to a molecular fragment
    """
    def __init__(self, prims, molecule, attached=[], coeff=1):
        self.prims = prims 
        self.molecule = molecule 
        self.coeff = coeff 
        self.attached = attached
        self.inputxyz = str()
        self.energy = 1

    def __str__(self):
        out = "Frag:"
        out += str(self.prims)
        out += str(self.coeff)
        out += str(self.attached)
        return out 

    def __repr__(self):
        return str(self)

    def build_xyz(self):    #builds input with atom label, xyz coords, and link atoms
        for atom in self.prims:
            atom_xyz = str(self.molecule.atomtable[atom]).replace('[', '')
            atom_xyz = atom_xyz.replace(']', '\n')
            atom_xyz = atom_xyz.replace(',', '')   
            self.inputxyz += atom_xyz
        for pair in self.attached:
            linkatom_xyz = str(add_linkatoms(pair[0], pair[1], self.molecule)).replace('[', '')
            linkatom_xyz = linkatom_xyz.replace(']', '\n')
            linkatom_xyz = linkatom_xyz.replace(',', '')   
            self.inputxyz += linkatom_xyz
    
    def run_pyscf(self):
        self.energy = do_pyscf(self.inputxyz)
        #print(self.energy)

