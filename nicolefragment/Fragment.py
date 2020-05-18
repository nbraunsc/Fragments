from runpyscf import *
import string
import numpy as np
#import torch
#import autograd.numpy as np
#from autograd import grad as a_grad
#from autograd import hessian as a_hess


class Fragment():
    """
    Class to store a list of primitives corresponding to a molecular fragment
    """
    def __init__(self, theory, basis, prims, molecule, attached=[], coeff=1):
        self.prims = prims 
        self.molecule = molecule 
        self.coeff = coeff 
        self.attached = attached    #[(supporting, host), (supporting, host), ...]
        self.energy = 1
        self.grad_dict = {}#dictonary for atom gradients in prim after link atom projections, no link atoms included
        self.hess_dict = {}
        self.grad = []
        self.hess = []
        self.theory = theory
        self.basis = basis
        self.notes = []     # [index of link atom, factor, supporting atom, host atom]
        self.jacobian_grad = [] #array for gradient link atom projections
        self.jacobian_hess = []  #ndarray shape of full system*3 x fragment(with LA)*3

    def __str__(self):
        out = "Frag:"
        out += str(self.prims)
        out += str(self.coeff)
        out += str(self.attached)
        return out 

    def __repr__(self):
        return str(self)

    def add_linkatoms(self, atom1, attached_atom, molecule):
        """
        Adds H as a link atom at a distance ratio between the supporting and host atom to each fragment where a previous atom was cut
        :supporting atom is the one in the fragment
        :host atom is the one that was cut during fragmention
        :returns the new xyx of link atom and the factor used to pick the position
        """
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
        """
        Builds the xyz input with the atom labels, xyz coords, and link atoms as a string
        :returns the inputxyz as a string
        :returns a list called self.notes that has the form [index of link atom, factor, supporting atom, host atom]
        """
        atomxyz = str()   #makes sure strings are empty, unsure if I need this
        inputxyz = str()
        self.notes = []
        for atom in self.prims:
            atom_xyz = str(self.molecule.atomtable[atom]).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
            atomxyz += atom_xyz
            inputxyz += atom_xyz
        for pair in range(0, len(self.attached)):
            linkatom_xyz = str(self.add_linkatoms(self.attached[pair][0], self.attached[pair][1], self.molecule)[0]).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
            factor = self.add_linkatoms(self.attached[pair][0], self.attached[pair][1], self.molecule)[1]
            inputxyz += linkatom_xyz
            position = len(self.prims) + pair
            self.notes.append([position])
            self.notes[-1].append(factor)
            self.notes[-1].append(self.attached[pair][0])
            self.notes[-1].append(self.attached[pair][1])
        return inputxyz

    def run_pyscf(self, theory, basis):
        """
        Runs pyscf and returns the fragments energy and gradient
        :theory - theory for pyscf wanted, right now only 'RHF' and MP2 implemented
        :basis - basis set for pyscf
        """
        inputxyz = self.build_xyz()
        self.energy, self.grad, self.hess  = do_pyscf(inputxyz, self.theory, self.basis, hess=False)
        self.build_jacobian_Grad()
        self.grad = self.jacobian_grad.dot(self.grad)
        return self.grad

    def build_jacobian_Grad(self):
        """Builds Jacobians for gradient link atom projections
        :entries are floats and not matrices
        """
        array = np.zeros((self.molecule.natoms, len(self.prims)))
        linkarray = np.zeros((self.molecule.natoms, len(self.notes)))
        for i in range(0, len(self.prims)):
            array[self.prims[i]][i] = 1
        for j in range(0, len(self.notes)):
            factor = 1 - self.notes[j][1]
            linkarray[self.notes[j][2]][j] = factor
            linkarray[self.notes[j][3]][j] = self.notes[j][1]
        self.jacobian_grad = np.concatenate((array, linkarray), axis=1)
    
    def build_jacobian_Hess(self, shape):
        """
        Builds Jacobian matrix that is full system x subsystem.
        :entires are matrices
        :will be used for Hessian link atom projections
        """
        zero_list = []
        full_array = np.zeros((self.molecule.natoms, shape, 3, 3))

        for i in range(0, len(self.prims)):
            full_array[self.prims[i], i] = np.identity(3)
        for j in range(0, len(self.notes)):
            factor_s = 1-self.notes[j][1]
            factor_h = self.notes[j][1]
            x = np.zeros((3,3))
            np.fill_diagonal(x, factor_s)
            position = len(self.prims) + j
            full_array[self.notes[j][2]][position] = x
            np.fill_diagonal(x, factor_h)
            full_array[self.notes[j][3]][position] = x
        self.jacobian_hess = full_array

    def example_func(self, y):
        return y

    def do_Hessian(self):   #"Need to work on dimesions for the matrix multiplication"
        #self.hess = 0 #just to make sure it is zero to start 
        inputxyz = self.build_xyz()
        #self.hess = do_pyscf(inputxyz, self.theory, self.basis, hess=True)[2]
        hessian_example = a_hess(self.example_func)
        x = hessian_example(self.grad).diagonal(axis1=1, axis2=2)
        print(self.grad, 'self.grad')
        print(x, "hessian")
        print(x.shape, "hessian shape")
        
        #self.build_jacobian_Hess(self.hess.shape[0])
        #j_reshape = self.jacobian_hess.transpose(1,0,2, 3)
        #y = np.einsum('ijkl, jmln -> imkn', self.jacobian_hess, self.hess) 
        #self.hess = np.einsum('ijkl, jmln -> imkn', y, j_reshape) 

"""Stuff after this point is just the old link atom projection without using the Jacobian matrix"""
    
    #def old jacobian hessian build:
       # for l in range(0, self.molecule.natoms):    #making zero arrays len of full system
       #     zeros = np.zeros((3,3))
       #     zero_list.append(zeros)
       # 
       # for i in range(0, len(self.prims)): #adding in identity matrix in correct location (no link atoms yet)
       #     i_list = zero_list
       #     i_list[self.prims[i]] = np.identity(3)
       #     i_array = np.vstack(i_list) #len molecule x 3 array
       #     full_array.append(i_array)
       #     i_list[self.prims[i]] = np.zeros((3,3)) #chaning back to all zeros
       # 
       # for j in range(0, len(self.notes)): #adding in factors on diag for support and host atoms
       #     j_list = zero_list
       #     factor_s = 1-self.notes[j][1]   #support factor on diag
       #     factor_h = self.notes[j][1]     #host factor on diag
       #     a = j_list[self.notes[j][2]]
       #     np.fill_diagonal(a, factor_s)   #fillin with support factor
       #     j_list[self.notes[j][2]] = a
       #     b = j_list[self.notes[j][3]]
       #     np.fill_diagonal(b, factor_h)   #filling with host factor
       #     j_list[self.notes[j][3]] = b
       #     j_array = np.vstack(j_list)
       #     full_array.append(j_array)
       # 
       # self.jacobian_hess = np.concatenate(full_array, axis=1)
    
    #def distribute_linkgrad(self):     ####Old way of doing the gradient link atom projections###########
    #    """
    #    Projects link atom gradients back to its respective atoms (both supporting and host atoms)
    #    :returns a dictonary with atom indexes as the keys and the corresponding gradient for each atom is stored
    #    """
    #    for i in range(0, len(self.prims)):
    #        self.grad_dict[self.prims[i]] = self.grad[i]
    #    for j in self.notes:
    #        self.grad_dict[j[3]] = self.grad[int(j[0])]*j[1]    #link to ghost
    #        old_grad = self.grad_dict[j[2]]
    #        self.grad_dict[j[2]] = old_grad + self.grad[int(j[0])]*(1-j[1]) #link to real

        #def distribute_linkhessian(self):
    #    """Projects mass-weighted Hessian matrix elements of link atoms back to its respective atoms (both supporting and host atoms)
    #    :WORK IN PROGRESS
    #    """
    #    ghost_list = []
    #    print(len(self.hess), 'length of fragment hessian')
    #    print(len(self.notes), 'number of link atoms')
    #    print(len(self.prims), 'number of atoms in fragment')
    #    for i in range(0, len(self.prims)):
    #        self.hess_dict[self.prims[i]] = self.hess[i]
    #        print(self.hess[i], 'indiviudal atom hess')
    #    for j in self.notes:
    #        ghost_list.append(j[3])
    #        self.hess_dict[j[3]] = self.hess[int(j[0])]*(j[1]**2)   #link to ghost
    #        old_hess = self.hess_dict[j[2]]
    #        self.hess_dict[j[2]] = old_hess + self.hess[int(j[0])]*((1-j[1])**2)    #link to real
    #    #After i project, I need to insert np.zeros(3,3) at indices not in the fragment, but that are in full molecule
    #    for k in range(0, len(self.molecule.atomtable)):
    #        pass
