import string
import time
import numpy as np
#from .Pyscf import *
#from ase import Atoms
#from ase.calculators.vasp import Vasp
#from ase.vibrations import Infrared  
from numpy import linalg as LA 
from mendeleev import element

#import nicolefragment
#from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf

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
    
    def __init__(self, qc_class, molecule, prims, attached=[], coeff=1, step_size=0.001, local_coeff=1):
        self.prims = prims 
        self.molecule = molecule 
        self.coeff = coeff 
        self.attached = attached    #[(supporting, host), (supporting, host), ...]
        self.inputxyz = []
        self.apt = []
        self.aptgrad = np.array([])
        self.step = step_size
        self.energy = 0
        #self.grad = []
        self.grad = 0
        self.hessian = 0
        #self.hessian = []
        self.hess = []
        self.notes = []     # [index of link atom, factor, supporting atom, host atom]
        self.jacobian_grad = [] #array for gradient link atom projections
        self.jacobian_hess = []  #ndarray shape of full system*3 x fragment(with LA)*3
        self.qc_class = qc_class
        self.step_size = step_size
        self.local_coeff = local_coeff
        self.M = [] #this is the mass matrix for massweighting shape: (3N, 3N)
        
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
        return coord, factor
        #return new_xyz, factor, coord
    
    def build_xyz(self):
        """ Builds the xyz input with the atom labels, xyz coords, and link atoms as a string or list 
        
        Parameters
        ----------
        none
        
        Returns
        -------
        inputxyz : str
            String with atom label then corresonding xyz coordinates.  This input includes the link atoms.
        input_list : list of lists
            ie [[['H', [0, 0 ,0]], ['O', [x, y, z]], ... ] 
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
        #self.input_list = input_list
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

    def qc_backend(self):
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
        np.set_printoptions(suppress=True, precision=7)
        self.energy = 0
        hess_py = 0
        #self.grad = np.zeros((self.molecule.natoms, 3))
        self.grad = 0
        self.inputxyz = self.build_xyz()
        energy, grad, hess_py = self.qc_class.energy_gradient(self.inputxyz)
        
        self.energy = self.local_coeff*self.coeff*energy
        jacob = self.build_jacobian_Grad()
        self.grad = self.local_coeff*self.coeff*jacob.dot(grad)
        self.M = self.mass_matrix()
        print("Done!")
        return self.energy, self.grad, hess_py  #, self.hessian#, self.apt

    def hess_apt(self, hess_py):
        print("Started hess_apt()")
        #If not analytical hess, do numerical below
        hess = hess_py
        if type(hess_py) is int:
            print("Numerical hessian needed, Theory=", self.qc_class.theory)
            hess = np.zeros(((len(self.inputxyz))*3, (len(self.inputxyz))*3))
            i = -1
            for atom in range(0, len(self.inputxyz)):
                for xyz in range(0, 3):
                    i = i+1
                    self.inputxyz[atom][1][xyz] = self.inputxyz[atom][1][xyz]+self.step_size
                    grad1 = self.qc_class.energy_gradient(self.inputxyz)[1].flatten()
                    self.inputxyz[atom][1][xyz] = self.inputxyz[atom][1][xyz]-2*self.step_size
                    grad2 = self.qc_class.energy_gradient(self.inputxyz)[1].flatten()
                    self.inputxyz[atom][1][xyz] = self.inputxyz[atom][1][xyz]+self.step_size
                    vec = (grad1 - grad2)/(4*self.step_size)
                    hess[i] = vec
                    hess[:,i] = vec
       
        hess = hess.reshape((len(self.prims)+len(self.notes), 3, len(self.prims)+len(self.notes), 3))
        hess = hess.transpose(0, 2, 1, 3)
        self.jacobian_hess = self.build_jacobian_Hess() #shape: (Full, Sub, 3, 3)
        
        hess_flat = hess.reshape(len(self.inputxyz)*3, len(self.inputxyz)*3, order='C') #shape: (Sub*3, Sub*3)
        j_reshape = self.jacobian_hess.transpose(0,2,1,3)   
        j_flat = j_reshape.reshape(self.molecule.natoms*3, len(self.inputxyz)*3)    #shape: (Full*3, Sub*3)
        j_flat_tran = j_flat.T  #shape: (Sub*3, Full*3)
        
        first = np.dot(j_flat, hess_flat)   # (Full*3, Sub*3) x (Sub*3, Sub*3) -> (Full*3, Sub*3)
        second = np.dot(first, j_flat_tran)     # (Full*3, Sub*3) x (Sub*3, Full*3) -> (Full*3, Full*3)
        self.hessian = second*self.coeff*self.local_coeff
        print(self.hessian.shape, "self.hessian.shape")

        self.apt = self.build_apt()    #one that words sorta
        self.aptgrad = self.apt_grad()     #one i am trying to get to work
        
        #build frag_hess, do link atom projection for hessian
        #j_reshape_old = self.jacobian_hess.transpose(1,0,2, 3)  #current one that works 
        #y = np.einsum('ijkl, jmln -> imkn', self.jacobian_hess, hess) 
        #old_hessian = np.einsum('ijkl, jmln -> imkn', y, j_reshape_old)*self.coeff*self.local_coeff
        
        #start of me trying to undo tensor stuff
        return self.hessian, self.apt
        
    def apt_grad(self):
        """ Working on implementing this.
        Function to create the apts by applying an electric field in a certain direciton to 
        molecule then centeral difference with gradient after E field is applied.

        !!! need to make sure h_core is getting changed for grad calc !!!
        """
        e_field = 1e-3
        E = [0, 0, 0]
        energy_vec = np.zeros((3))
        apt = np.zeros((3, ((len(self.prims)+len(self.notes))*3)))
        grad_list = []
        extra_dip = self.qc_class.get_dipole(self.inputxyz)
        print(extra_dip)
        for i in range(0, 3):
            #e1, g1, dip = self.qc_class.apply_field(E, self.inputxyz)   #no field
            E[i] = e_field
            #print("energy with no field:", e1)
            print("\n############ Field applied in the ", i, "direction ###############\n")
            #print("electric field positive:", E)
            e2, g2, dipole2 = self.qc_class.apply_field(E, self.inputxyz) #positive direction
            print("gradient positive:", g2)
            #print("energy with + field:", e2)
            E[i] = -1*e_field
            #print("electric field negative:", E)
            e3, g3, dipole3 = self.qc_class.apply_field(E, self.inputxyz)   #neg direction
            print("gradient neg:", g3)
            #print("energy with - field:", e3)
            E[i] = 0
            #print("should be back to 0:", E)
    
            #print("energy differnce between + and -:", e2-e3, "for direction:", i)
            #print("Energy difference:", e2-e1)
            #gradient = (g2.flatten()-g3.flatten())/(2*e_field)
            gradient1 = (g2-g3)/(2*e_field)
            gradient = gradient1.flatten()
            energy2 = (e2-e3)/(2*e_field)
            #print("derv energy comp:", energy2)
            energy_vec[i] = energy2
            print("energy vec:", energy_vec)
            print("Dipole moment:\n", extra_dip)
            print("gradient shape:", gradient, gradient.shape)
            apt[i] = gradient
        
        print(apt, apt.shape)
        print("fragment:", self.prims)
        #print("norm of E\n", np.linalg.norm(energy_vec))
        #print("norm of dip\n", np.linalg.norm(dip))
        
        #for atom in range(0, len(self.prims)+len(self.notes)):  #atom interation
        #    y = element(self.inputxyz[atom][0])
        #    value = 1/(np.sqrt(y.atomic_weight))


        ##build M^-1/2 mass matrix and mass weight apt
        #labels = []
        #mass_matrix = np.zeros((apt.shape[1], apt.shape[1]))
        #for i in range(0, len(self.prims)):
        #    labels.append(self.molecule.atomtable[self.prims[i]][0])
        #    labels.append(self.molecule.atomtable[self.prims[i]][0])
        #    labels.append(self.molecule.atomtable[self.prims[i]][0])
        #for pair in range(0, len(self.attached)):
        #    labels.append('H')
        #    labels.append('H')
        #    labels.append('H')
        #for atom in range(0, len(labels)):
        #    y = element(labels[atom])
        #    value = 1/(np.sqrt(y.atomic_weight))
        #    mass_matrix[atom][atom] = value
        #apt_mass = np.dot(mass_matrix, apt.T)
        #print(apt_mass)
        #print("\n")
        #print(self.apt)
        
        reshape_mass_hess = self.jacobian_hess.transpose(0, 2, 1, 3)
        jac_apt = reshape_mass_hess.reshape(reshape_mass_hess.shape[0]*reshape_mass_hess.shape[1],reshape_mass_hess.shape[2]*reshape_mass_hess.shape[3])
        apt_grad = self.local_coeff*self.coeff*np.dot(jac_apt, apt.T)
        #apt_grad = self.local_coeff*self.coeff*np.dot(jac_apt, apt_mass)
        return apt_grad
            
    def build_apt(self):
        """
            Builds the atomic polar tensor with numerical derivative of dipole moment w.r.t atomic Cartesian
            coordinates. Function builds xyz input with link atoms in ndarray format, not string type or list like previous functions.

        Units of APT: Debye**2 / (Angstrom**2 amu***1)

        """
        print("input coords", self.inputxyz)
        apt = []
        for atom in range(0, len(self.prims)+len(self.notes)):  #atom interation
            storing_vec = np.zeros((3,3))
            y = element(self.inputxyz[atom][0])
            value = 1/(np.sqrt(y.atomic_weight))
            for comp in range(0,3):   #xyz interation
                self.inputxyz[atom][1][comp] = self.inputxyz[atom][1][comp]+self.step_size
                print("\n", self.inputxyz[atom][1][comp], self.inputxyz[atom], "\n")
                dip1 = self.qc_class.get_dipole(self.inputxyz)
                self.inputxyz[atom][1][comp] = self.inputxyz[atom][1][comp]-2*self.step_size
                dip2 = self.qc_class.get_dipole(self.inputxyz)
                vec = (dip1 - dip2)/(2*self.step_size)
                print("\n Vector for derivative:", vec, vec.shape)
                print("#######")
                storing_vec[comp] = vec
                self.inputxyz[atom][1][comp] = self.inputxyz[atom][1][comp]+self.step_size
            print(storing_vec, "before mass weighting")
            print("atom:", self.inputxyz[atom][0])
            a = storing_vec*value    ##mass weighting
            print("element and value:", y, value)
            print(a, a.shape, "after mass weighting")
            print("#######")
            apt.append(a)
            print(apt)
        px = np.vstack(apt)
        print(px, px.shape)
        print("\n")
        reshape_mass_hess = self.jacobian_hess.transpose(0, 2, 1, 3)
        jac_apt = reshape_mass_hess.reshape(reshape_mass_hess.shape[0]*reshape_mass_hess.shape[1],reshape_mass_hess.shape[2]*reshape_mass_hess.shape[3])
        oldapt = self.local_coeff*self.coeff*np.dot(jac_apt, px)
        return oldapt

    def mass_matrix(self):
        M = np.zeros((self.molecule.natoms*3, self.molecule.natoms*3))
        counter = np.array([0, 1, 2])
        for i in range(0, self.molecule.natoms):
            x = element(self.molecule.atomtable[i][0])
            value = 1/(np.sqrt(x.atomic_weight))
            for j in counter:
                M[j][j] = value
            counter = counter + 3
        self.M = M
        return self.M

    def mw_hessian(self, full_hessian):
        """
        Will compute the mass-weighted hessian, frequencies, and 
        normal modes for the full system.
        
        Parameters
        ----------
        full_hessian : ndarray
            This is the full hessian for the full molecule.

        Returns
        -------
        freq : ndarray
            1D np array holding the frequencies
        modes : ndarray
            2D ndarray holding normal modes in the columns
        """
        np.set_printoptions(suppress=True)
        first = np.dot(full_hessian, self.M) #shape (3N,3N) x (3N, 3N)
        second = np.dot(self.M, first)   #shape (3N,3N) x (3N, 3N)
        e_values, modes = LA.eigh(second)

        #unit conversion of freq from H/A**2 amu -> 1/s**2
        #factor = (4.3597482*10**-18)/(1.6603145*10**-27)/(1.0*10**-20)  #Angstrom to m
        factor = 1.8897259886**2*(4.3597482*10**-18)/(1.6603145*10**-27)/(1.0*10**-20) #Bohr to Angstrom
        freq = (np.sqrt(e_values*factor))/(2*np.pi*2.9979*10**10) #1/s^2 -> cm-1
        return freq, modes, self.M

#everything following works perfectly but only for water but freq and intensities match
       # full_hessian = full_hessian.transpose(0, 2, 1, 3)
       # for row in range(0, self.molecule.natoms):
       #     #adding 1/np.sqrt(amu) units to xyz coords
       #     x = element(self.molecule.atomtable[row][0])
       #     value = np.sqrt(x.atomic_weight)
       #     #mass_array[row] = value
       #     #mass_xyz[row] = np.array(self.molecule.atomtable[row][1:])*(1/value)   #mass weighted coordinates

       #     for column in range(0, self.molecule.natoms):    
       #         y = element(self.molecule.atomtable[column][0])
       #         z = x.atomic_weight*y.atomic_weight
       #         value = np.sqrt(z)**-1
       #         #print(full_hessian[row][column])
       #         #term = full_hessian[row, :, column, :]*value
       #         #full_hessian[row, :, column, :] = term
       #         
       #         term = full_hessian[row][column]*value
       #         full_hessian[row][column] = term
       #         full_hessian[column][row] = term.T
       # print(full_hessian) 
       # #reshape_mass_hess = full_hessian.transpose(0, 2, 1, 3)
       # reshape_mass_hess = full_hessian
       # x = reshape_mass_hess.reshape(reshape_mass_hess.shape[0]*reshape_mass_hess.shape[2],reshape_mass_hess.shape[1]*reshape_mass_hess.shape[3])
       # #x = reshape_mass_hess.reshape(reshape_mass_hess.shape[0]*reshape_mass_hess.shape[2],reshape_mass_hess.shape[1]*reshape_mass_hess.shape[3])
       # e_values, modes = LA.eigh(x)

       # #unit conversion of freq from H/B**2 amu -> 1/s**2
       # #factor = (4.3597482*10**-18)/(1.6603145*10**-27)/(1.0*10**-20)  #Angstrom to m
       # factor = 1.8897259886**2*(4.3597482*10**-18)/(1.6603145*10**-27)/(1.0*10**-20) #Bohr to Angstrom
       # freq = (np.sqrt(e_values*factor))/(2*np.pi*2.9979*10**10) #1/s^2 -> cm-1
       # return freq, modes

