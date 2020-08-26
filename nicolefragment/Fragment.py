import string
import numpy as np
from .Pyscf import *
from ase import Atoms
from ase.vibrations import Infrared  
from mendeleev import element

class Fragment():
    """
    Class to store a list of primitives corresponding to a molecular fragment
    
    Parameters
    ----------
    theory : str
        Level of theory for calculation
    basis : str
        Basis set name for calculations
    prims : list
        List of fragments from Fragmentation class with atom indexes in list
    attached : list
        List of attached pairs with the atom that is in the fragment and its corresponding atom pair that was cut
    coeff : int
        Coefficent of fragment.  This will either be 1 or -1.
    
    """
    
    def __init__(self, qc_class, molecule, prims, attached=[], coeff=1, step=0.001):
        self.prims = prims 
        self.molecule = molecule 
        self.coeff = coeff 
        self.attached = attached    #[(supporting, host), (supporting, host), ...]
        self.apt = []
        self.step = step
        self.energy = 1
        self.grad = []
        self.hessian = []
        self.hess = []
        self.notes = []     # [index of link atom, factor, supporting atom, host atom]
        self.jacobian_grad = [] #array for gradient link atom projections
        self.jacobian_hess = []  #ndarray shape of full system*3 x fragment(with LA)*3
        self.qc_class = qc_class
        

    #def __str__(self):
    #    out = "Frag:"
    #    out += str(self.prims)
    #    out += str(self.coeff)
    #    out += str(self.attached)
    #    return out 

    #def __repr__(self):
    #    return str(self)

    def add_linkatoms(self, atom1, attached_atom, molecule):
        """ Adds H as a link atom
        
        This link atoms adds at a distance ratio between the supporting and host atom to each fragment where a previous atom was cut
        
        Parameters
        ----------
        atom1 : int
            This is the integer corresponding to the supporting atom (real atom)
        attached_atom : int
            This is the integer corresponiding to the host atom (ghost atom)
        molecule : <class> instance
            This is the molecule class instance

        Returns
        -------
        new_xyz : list
            This is the list of the new link atom with atom label and xyz coords
        factor : float
            The factor between the supporting and host atom. Used in building Jacobians for link atom projections.

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
        coord = []
        coord.append('H')
        coord.append(new_xyz)
        #new_xyz.insert(0, 'H')
        return coord, factor
        #return new_xyz, factor, coord
    
    def build_xyz(self):    #builds input with atom label, xyz coords, and link atoms as a string
        """ Builds the xyz input with the atom labels, xyz coords, and link atoms as a string
        
        Parameters
        ----------
        none
        
        Returns
        -------
        inputxyz : str
            String with atom label then corresonding xyz coordinates.  This input includes the link atoms.
        self.notes: list of lists
            List of lists that is created with len = number of link atoms. Each sub list corresponds to one link atom.
            (i.e. [index of link atom, factor, supporting atom number, host atom number])
        
        """

        #atomxyz = str()   #makes sure strings are empty, unsure if I need this
        #inputxyz = str()
        #self.notes = []
        #for atom in self.prims:
        #    atom_xyz = str(self.molecule.atomtable[atom]).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
        #    atomxyz += atom_xyz
        #    inputxyz += atom_xyz
        #for pair in range(0, len(self.attached)):
        #    linkatom_xyz = str(self.add_linkatoms(self.attached[pair][0], self.attached[pair][1], self.molecule)[0]).replace('[', '').replace(']', '\n').replace(',', '').replace("'", "")
        #    factor = self.add_linkatoms(self.attached[pair][0], self.attached[pair][1], self.molecule)[1]
        #    inputxyz += linkatom_xyz
        #    position = len(self.prims) + pair
        #    self.notes.append([position])
        #    self.notes[-1].append(factor)
        #    self.notes[-1].append(self.attached[pair][0])
        #    self.notes[-1].append(self.attached[pair][1])
        self.notes = []
        input_list = []
        coord_matrix = np.empty([len(self.prims)+len(self.attached), 3])
        for atom in self.prims:
            input_list.append([self.molecule.atomtable[atom][0]])
            input_list[-1].append(list(self.molecule.atomtable[atom][1:]))
            x = np.array(self.molecule.atomtable[atom][1:])
        for pair in range(0, len(self.attached)):
            la_input, factor = self.add_linkatoms(self.attached[pair][0], self.attached[pair][1], self.molecule)
            input_list.append(la_input)
            position = len(self.prims)+pair
            self.notes.append([position])
            self.notes[-1].append(factor)
            self.notes[-1].append(self.attached[pair][0])
            self.notes[-1].append(self.attached[pair][1])
        return input_list
    
    def build_jacobian_Grad(self):
        """Builds Jacobian matrix for gradient link atom projections
        
        Parameters
        ----------
        none
        
        Returns
        -------
        self.jacobian_grad : ndarray
            Array where entries are floats on the diagonal with the corresponding factor. 
            Array has size (# of atoms in full molecule + all link atoms, # of atoms in primiative)
        
        """
        self.jacobian_grad = 0
        array = np.zeros((self.molecule.natoms, len(self.prims)))
        linkarray = np.zeros((self.molecule.natoms, len(self.notes)))
        for i in range(0, len(self.prims)):
            array[self.prims[i]][i] = 1
        for j in range(0, len(self.notes)):
            factor = 1 - self.notes[j][1]
            linkarray[self.notes[j][2]][j] = factor
            linkarray[self.notes[j][3]][j] = self.notes[j][1]
        self.jacobian_grad = np.concatenate((array, linkarray), axis=1)
        jacob = self.jacobian_grad
        return jacob
    
    def build_jacobian_Hess(self):
        """ Builds Jacobian matrix for hessian link atom projections.

        Parameters
        ----------

        Returns
        -------
        self.jacobian_hess : ndarray (tensor)
            Array where the entries are matrices corresponding factor.
            
        """
        
        zero_list = []
        full_array = np.zeros((self.molecule.natoms, len(self.prims)+len(self.notes), 3, 3))

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
        return self.jacobian_hess

    def qc_backend(self, step=0.001):
        """ Runs the quantum chemistry backend.
        
        Returns
        -------
        self.energy : float
            This is the energy for the fragment*its coeff
        self.gradient : ndarray
            This is the gradient for the fragment*its coeff
        self.hessian : ndarray (4D tensor)
            This is the hessian for the fragement*its coeff
        """
        np.set_printoptions(suppress=True, precision=5)
        self.energy = 0
        #hess_py = np.zeros(((len(self.prims)+len(self.notes))*3, (len(self.prims)+len(self.notes))*3))
        hess_py = 0
        self.grad = np.zeros((self.molecule.natoms, 3))
        inputxyz = self.build_xyz()
        energy, grad, hess_py = self.qc_class.energy_gradient(inputxyz)
        hess = hess_py

        #If not analytical hess, not do numerical below
        if type(hess_py) is int:
            hess = np.zeros(((len(inputxyz))*3, (len(inputxyz))*3))
            i = -1
            for atom in range(0, len(inputxyz)):
                for xyz in range(0, 3):
                    i = i+1
                    inputxyz[atom][1][xyz] = inputxyz[atom][1][xyz]+step
                    grad1 = self.qc_class.energy_gradient(inputxyz)[1].flatten()
                    inputxyz[atom][1][xyz] = inputxyz[atom][1][xyz]-2*step
                    grad2 = self.qc_class.energy_gradient(inputxyz)[1].flatten()
                    inputxyz[atom][1][xyz] = inputxyz[atom][1][xyz]+step
                    vec = (grad1 - grad2)/(4*step)
                    hess[i] = vec
                    hess[:,i] = vec
       
            hess = hess.reshape((len(self.prims)+len(self.notes), 3, len(self.prims)+len(self.notes), 3))
            hess = hess.transpose(0, 2, 1, 3)
        
        self.energy = self.coeff*energy
        jacob = self.build_jacobian_Grad()
        self.grad = self.coeff*jacob.dot(grad)
        
        #build frag_hess, do link atom projection for hessian
        self.jacobian_hess = self.build_jacobian_Hess()
        j_reshape = self.jacobian_hess.transpose(1,0,2, 3)
        y = np.einsum('ijkl, jmln -> imkn', self.jacobian_hess, hess) 
        self.hessian = np.einsum('ijkl, jmln -> imkn', y, j_reshape)*self.coeff
        self.apt = self.build_apt()
        return self.energy, self.grad, self.hessian, self.apt
    
    def build_apt(self):
        #build xyz with link atoms in ndarray format, not string type like function
        x = np.zeros((len(self.prims)+len(self.notes), 3))
        labels = []
        for i in range(0, len(self.prims)):
            x[i] = (self.molecule.atomtable[self.prims[i]][1:])
            labels.append(self.molecule.atomtable[self.prims[i]][0])
        
        for pair in range(0, len(self.attached)):
            linkatom_xyz = self.add_linkatoms(self.attached[pair][0], self.attached[pair][1], self.molecule)[0]
            x[self.notes[pair][0]] = linkatom_xyz[1]
            #x[self.notes[pair][0]] = linkatom_xyz[1:]
            labels.append(linkatom_xyz[0])
        
        #adding 1/np.sqrt(amu) units to xyz coords
        mass_xyz = np.zeros(x.shape)
        for atom in range(0, len(self.prims)+len(self.notes)):
            y = element(labels[atom])
            value = np.sqrt(y.atomic_weight)
            mass_xyz[atom] = x[atom]*(1/value)   #mass weighted coordinates
        
        #formatting xyz for pyscf coords input
        coords_xyz = []
        for atom in range(0, len(labels)):
            coords_xyz.append([labels[atom], x[atom]])
            #coords_xyz.append([labels[atom], mass_xyz[atom]])
        
        apt = []
        for atom in range(0, len(self.prims)+len(self.notes)):  #atom interation
            storing_vec = np.zeros((3,3))
            y = element(labels[atom])
            value = 1/(np.sqrt(y.atomic_weight))
            for comp in range(0,3):   #xyz interation
                dip1 = self.qc_class.get_dipole(coords_xyz)
                coords_xyz[atom][1][comp] = coords_xyz[atom][1][comp]+self.step
                dip2 = self.qc_class.get_dipole(coords_xyz)
                vec = (dip1 - dip2)/self.step
                storing_vec[comp] = vec
                coords_xyz[atom][1][comp] = coords_xyz[atom][1][comp]-self.step
            a = storing_vec.T*value
            apt.append(a)
        px = np.vstack(apt)
        reshape_mass_hess = self.jacobian_hess.transpose(0, 2, 1, 3)
        jac_apt = reshape_mass_hess.reshape(reshape_mass_hess.shape[0]*reshape_mass_hess.shape[1],reshape_mass_hess.shape[2]*reshape_mass_hess.shape[3])
        self.apt = self.coeff*np.dot(jac_apt, px)
        return self.apt
        
        #############
        # stuff below is for the intensity stuff with normal modes
        #############

        #intensity = []
        #pq = np.dot(px.T, modes)
        #print("pq = ", pq.shape, pq)
        #pq_pq = np.dot(pq.T, pq)
        #print(pq_pq.shape)
        #intense = np.diagonal(pq_pq)
        #print("intensities in other units", intense)
        #intense_kmmol = intense*42.2561*0.529177
        #print("intensities in km/mol", intense_kmmol)
        #webmo_int = [77.298, 19.019, 51.071]
        #print('Webmo intensites', webmo_int)
        #diff_int = np.array(intense_kmmol[6:]) - webmo_int
        #factor_int = webmo_int/np.array(intense_kmmol[6:])
        #print("apt int factors", factor_int)
        #print("intensity diff", diff_int)
        
    
    def get_IR(self):
        inputxyz = self.build_xyz()
        print(inputxyz, type(inputxyz))
        molecule = Atoms(inputxyz)
        print(molecule.get_positions())
        ir = Infrared(molecule)
        ir.run()
        ir.summary()







####################################################################################


    def run_pyscf(self, theory, basis):
        """ Runs pyscf.  This gets called in Fragmentation().

        do_pyscf() is just a virtual funciton in runpysf.py that interfaces with pyscf.
        
        Parameters
        ----------
        theory : str
            Level of theory for the calculation
        basis : str
            Basis set name for calculations
            
        Returns
        -------
        self.energy : float
            Energy for the individual fragment after link atom porjection
        self.grad : ndarray
            Gradient for the individual fragment after link atom projection
        
        """
        
        inputxyz = self.build_xyz()
        self.energy, self.grad, self.hess  = do_pyscf(inputxyz, self.theory, self.basis, hess=False)
        self.build_jacobian_Grad()
        self.grad = self.jacobian_grad.dot(self.grad)
        return self.grad

    def run_psi4(self, theory, basis, name):
        """ Runs psi4.  This gets called in Fragmentation().

        do_psi4() is just a virtual funciton in runpsi4.py that interfaces with psi4.
        
        Parameters
        ----------
        theory : str
            Level of theory for the calculation
        basis : str
            Basis set name for calculations
            
        Returns
        -------
        self.energy : float
            Energy for the individual fragment after link atom porjection
        self.grad : ndarray
            Gradient for the individual fragment after link atom projection
        
        """
        
        inputxyz = self.build_xyz()
        self.energy = do_psi4(inputxyz, self.theory, self.basis, name)
        #self.energy, self.grad  = do_psi4(inputxyz, self.theory, self.basis, name)
        print("psi4 energy: ", self.energy)
        #print("psi4 grad: ", self.grad, self.grad.shape)
        #self.build_jacobian_Grad()
        #self.grad = self.jacobian_grad.dot(self.grad)
        #return self.grad
    
        

#-----------------------------------------------------------------------------------------------

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
