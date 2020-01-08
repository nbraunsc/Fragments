from linkatoms import *

class Fragment():
    """
    Class to store a list of primitives corresponding to a molecular fragment
    """
    def __init__(self, prims, molecule, attached=[], coeff=1):
        self.prims = prims 
        self.molecule = molecule 
        self.coeff = coeff 
        self.attached = attached
        self.inputxyz = []


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
            atom_xyz = self.molecule.atomtable[atom]
            self.inputxyz.append(atom_xyz)
        for pair in self.attached:
            linkatom_xyz = add_linkatoms(pair[0], pair[1], self.molecule)
            self.inputxyz.append(linkatom_xyz)
        

