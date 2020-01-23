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
        self.attached = attached    #[(supporting, host), (supporting, host), ...]
        self.atomxyz = str()
        self.inputxyz = str()
        self.energy = 1
        self.grad_dict = {}
        self.grad = []
        self.theory = theory
        self.basis = basis
        self.notes = []     # [position of link atom, factor, supporting atom, host atom]

    def __str__(self):
        out = "Frag:"
        out += str(self.prims)
        out += str(self.coeff)
        out += str(self.attached)
        return out 

    def __repr__(self):
        return str(self)

    def add_linkatoms(self, atom1, attached_atom, molecule):    #atom1 is supporting atom(one in fragment), attached_atom is host atom(one that was cut during fragmentation)
        atom1_element = molecule.atomtable[atom1][0]
        attached_atom_element = molecule.atomtable[attached_atom][0]
        cov_atom1 = molecule.covrad[atom1_element][0]
        cov_attached_atom = molecule.covrad[attached_atom_element][0]
        self.atom_xyz = np.array(molecule.atomtable[atom1][1:])
        attached_atom_xyz = np.array(molecule.atomtable[attached_atom][1:])
        vector = attached_atom_xyz - self.atom_xyz
        dist = np.linalg.norm(vector)
        h = 0.32
        factor = (h + cov_atom1)/(cov_atom1 + cov_attached_atom)
        new_xyz = list(factor*vector+self.atom_xyz)
        new_xyz.insert(0, 'H')
        return new_xyz, factor
    
    def build_xyz(self):    #builds input with atom label, xyz coords, and link atoms as a string
        for atom in self.prims:
            atom_xyz = str(self.molecule.atomtable[atom]).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
            self.atomxyz += atom_xyz
            self.inputxyz += atom_xyz
        for pair in range(0, len(self.attached)):
            linkatom_xyz = str(self.add_linkatoms(self.attached[pair][0], self.attached[pair][1], self.molecule)[0]).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
            factor = self.add_linkatoms(self.attached[pair][0], self.attached[pair][1], self.molecule)[1]
            self.inputxyz += linkatom_xyz
            position = len(self.prims) + pair
            self.notes.append([position])
            self.notes[-1].append(factor)
            self.notes[-1].append(self.attached[pair][0])
            self.notes[-1].append(self.attached[pair][1])
    
    def run_pyscf(self, theory, basis):
        print(self.prims)
        print(self.attached, 'attached')
        self.energy, self.grad = do_pyscf(self.inputxyz, self.theory, self.basis)
        #self.grad = list(self.grad) 
        #x = len(self.prims)
        #for i in range(x, len(self.grad)):
            #mag = np.linalg.norm(self.grad[i])

    def distribute_linkgrad(self):  #projects link-atom gradients back to respective atoms (supporting and host)
        for i in range(0, len(self.prims)):
            self.grad_dict[self.prims[i]] = self.grad[i]
        for j in self.notes:
            self.grad_dict[j[3]] = self.grad[int(j[0])]*j[1]
            old_grad = self.grad_dict[j[2]]
            self.grad_dict[j[2]] = old_grad + self.grad[int(j[0])]*(1-j[1])

                    


"""Made dictonary where the each atom has its label (0, 1, 2, ... number of atoms in molecule). Then assign to each atom the gradients found from each fragment.  Now I can add the gradients from the link atoms to each atom it was attached to based on the atom key verses the index in each fragment """
