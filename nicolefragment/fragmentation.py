import time
import os
import numpy as np
from numpy import linalg as LA 
import sys
from sys import argv
import xml.etree.ElementTree as ET
from itertools import cycle
from mendeleev import element

import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf

class Fragmentation():
    """
    Fragmentation is a child class of Molecule().
    
    Used to build the molecular fragments and their derivatives.
    
    An instance of this class represents a mapping between a large molecule and a list of independent fragments with the appropriate coefficients needed to reassemble expectation values of observables. 
    """
    
    def __init__(self, molecule): 
        self.fragment = []
        self.molecule = molecule
        self.unique_frag = []
        self.derivs = []
        self.coefflist = []
        self.atomlist = []
        self.frags = []     #list of Fragment class instances
        self.etot = 0
        self.gradient = []
        self.hessian = []
        self.fullgrad = {}  #dictonary for full molecule gradient
        self.fullhess = {}
        self.moleculexyz = []   #full molecule xyz's
        self.etot_opt = 0
        self.grad_opt = []

    def build_frags(self, frag_type=None, value=None):    #deg is the degree of monomers wanted
        """ Performs the initalize fragmmentation 
        
        This will fragment the full molecule at the level of deg specified.
        Also runs self.compress_frags() to delete repeating fragments.
        
        Parameters
        ----------
        frag_type : str
            This is the type of fragmentation wanted either distance or graphical.

        value : int
            The degree of fragmentation wanted. Larger deg results in larger fragments. For distance type it will
            be the largest radius cuttoff. For covalent_net it will be the number of bonds away from a center.
        
        Returns
        -------
        self.fragment : list
            List of fragments containing index of primiatives that are within the value
        
        """
        self.fragment = []
        if frag_type == 'graphical':
            self.molecule.molchart = self.molecule.build_molmatrix(2)
            for x in range(0, len(self.molecule.molchart)):
                for y in range(0, len(self.molecule.molchart)):
                    if self.molecule.molchart[x][y] <= value and self.molecule.molchart[x][y] != 0:
                        if x not in self.fragment:
                            self.fragment.append([x])
                            self.fragment[-1].append(y)
            for z in range(0, len(self.fragment)):
                for w in range(0, len(self.fragment)):
                    if z == w:
                        continue
                    if self.fragment[z][0] == self.fragment[w][0]:
                        self.fragment[z].extend(self.fragment[w][:])    #combines all prims with frag connectivity <= eta
            
        if frag_type == 'distance':
            self.molecule.prim_dist = self.molecule.build_prim_dist()
            for a in range(0, len(self.molecule.prims)):
                #prim = list(self.molecule.prims[a])
                arr = self.molecule.prim_dist[a]
                self.fragment.append([a])
                x = np.where(arr<=value)
                for i in x[0]:
                    if arr[i] != 0:
                        self.fragment[a].extend([i])
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

    def remove_repeatingfrags(self, oldcoeff):
        """ Removes the repeating derivates. Function gets called in self.do_fragmentation().
        
        The repeating derivates are the exact same derivates that would be added then subtracted
        again during the principle of inclusion-exculsion process of MIM.  This also updates the self.derivs
        and self.coefflist that are using with Fragment() class instances.
        
        Parameters
        ----------
        oldcoeff : list
            List of coefficients that is the output of runpie() function that is called in self.do_fragmentation().
        
        Returns
        -------v
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
    
    def initalize_Frag_objects(self, theory=None, basis=None, qc_backend=None, spin=None, tol=None, active_space=None, nelec=None, nelec_alpha=None, nelec_beta=None, max_memory=None, step_size=0.001):
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
        
        self.frags = [] #list of Fragment instances as ray Actor ids
        for fi in range(0, len(self.atomlist)):
            coeffi = self.coefflist[fi]
            attachedlist = self.attached[fi]
            qc_fi = qc_backend(theory=theory, basis=basis, spin=spin, tol=tol, active_space=active_space, nelec_alpha=nelec_alpha, nelec_beta=nelec_beta, max_memory=max_memory)
            self.frags.append(Fragment.Fragment(qc_fi, self.molecule, self.atomlist[fi], attachedlist, coeff=coeffi, step_size=step_size))
    
    def qc_params(self, frag_index=[], qc_backend=None, theory=None, basis=None, spin=0, tol=0, active_space=0, nelec=0, nelec_alpha=0, nelec_beta=0, max_memory=0):
        """ Funciton that is optional
        
        Use if a certain fragment or list of fragments need to be run with a different qc backend, or theory, or needs specific params.

        Parameters
        ----------
        frag_index : list
            The location of the specific fragment within the fragment list
        qc_backend : str
            Quantum chemistry backend wanted for this fragment
        theory : str
            Theory wanted
        basis : str
            Basis set wanted
        spin : int
            Spin state for fragment
        tol : int
            Tolerance wanted
        active_space : int
            Size of active space for CASSCF etc.
        nelec_alpha : int
            Number of alpha electrons
        nelec_beta : int
            Number of beta electrons
        max_memory : int
            Max memory for this calculation
        """
        for fi in frag_index:
            qc_fi = qc_backend(theory=theory, basis=basis, spin=spin, tol=tol, active_space=active_space, nelec_alpha=nelec_alpha, nelec_beta=nelec_beta, max_memory=max_memory)
            
    
    def energy_gradient(self, newcoords):
        """ Function returns total energy and gradient of global molecule.
        
        This function holds virtual functions for different chemical software.  A software
        must be implemented inorder to run this function.  
        
        Parameters
        ----------
        newcoords : npdarray
            Contains xyz coords for the full molecule. These get updated after each geometry optimization cycle.
        
        Returns
        -------
        self.etot : float
            Energy of the full molecule
        self.gradient : ndarray
            Gradient of the full molecule
        
        """

        #for atom in range(0, len(self.molecule.atomtable)): #makes newcoords = self.molecule.atomtable
        #    x = list(newcoords[atom])
        #    self.molecule.atomtable[atom][1:] = x


        self.gradient = np.zeros((self.molecule.natoms,3)) #setting them back to zero
        self.etot = 0
        self.hessian = np.zeros((self.molecule.natoms, self.molecule.natoms, 3, 3)) 

        for i in self.frags:
            #i.molecule.atomtable = self.molecule.atomtable  #setting newcoords
            e, grad, i_hess, i_hess2 = i.qc_backend()
            print(e, e.shape)
            print(grad, grad.shape)
            print(e-grad)
            print("hessian diff = ", i_hess-i_hess2)
            self.etot += i.energy
            self.gradient += i.grad
            self.hessian += i.hessian
        return self.etot, self.gradient, self.hessian
           
    def write_xyz(self, name):
        """ Writes an xyz file with the atom labels, number of atoms, and respective Cartesian coords for geom_opt().

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
        coords = molecule[:, [1,2,3]]
        print("coords thing", coords, coords.shape)
        self.moleculexyz = []
        for i in coords:
            #x = [i[1], i[2], i[3]]
            y = np.array(i)
            z = y.astype(float)
            self.moleculexyz.append(z)
        self.moleculexyz = np.array(self.moleculexyz)   #formatting full molecule coords
        
        f = open("../inputs/" + name + ".xyz", "w+")
        title = ""
        f.write("%d\n%s\n" % (self.moleculexyz.size / 3, title))
        for x, atomtype in zip(self.moleculexyz.reshape(-1, 3), cycle(atomlabels)): 
            f.write("%s %.18g %.18g %.18g\n" % (atomtype, x[0], x[1], x[2]))
        f.close()

    def do_fragmentation(self, frag_type=None, value=None):
        """ Main executeable for Fragmentation () class

        This function fragments the molecule, runs principle of inclusion-exculsion,
        removes repeating fragments, creates attached pairs list, initalizes instances of Fragment() class.
        
        Parameters
        ----------
        frag_type : str
            Either 'graphical' to do covalent network or 'radius' to do spacial fragmentation
        value : int
            Degree or radius of fragmentation wanted, gets used in self.build_frags()
        
        Returns
        -------
        none
        
        """
        self.build_frags(frag_type=frag_type, value=value)
        self.derivs, oldcoeff = runpie.runpie(self.unique_frag)
        self.remove_repeatingfrags(oldcoeff)
        self.atomlist = [None] * len(self.derivs)
        
        for i in range(0, len(self.derivs)):
            self.derivs[i] = list(self.derivs[i])
        for fragi in range(0, len(self.derivs)):    #changes prims into atoms
            x = len(self.derivs[fragi])
            for primi in range(0, x):
                y = self.derivs[fragi][primi]
                atoms = list(self.molecule.prims[y])
                self.derivs[fragi][primi] = atoms
        
        for y in range(0, len(self.derivs)):
            flatlist = [ item for elem in self.derivs[y] for item in elem]
            self.atomlist[y] = flatlist     #now fragments are list of atoms
        
        for i in range(0, len(self.atomlist)):  #sorted atom numbers
            self.atomlist[i] = list(sorted(self.atomlist[i]))
        self.find_attached()
        
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
        os.chdir('../inputs/')
        optimizer = Berny(geomlib.readfile(os.path.abspath(os.curdir) + '/' + name + '.xyz'), debug=True)
        for geom in optimizer:
            solver = self.energy_gradient(geom.coords)
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
       
   # def compute_Hessian(self):
   #     """
   #     Computes the overall Hessian for the molecule after the geometry optimization is completed.
   #     :Also does link atom projects for the Hessians
   #     """
   #     self.hessian = np.zeros((self.molecule.natoms, self.molecule.natoms, 3, 3)) 
   #     for i in self.frags:
   #         self.hessian += i.hess*i.coeff

   #     return self.hessian, hess_values, hess_vectors

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
        mass_xyz = np.zeros((self.molecule.natoms, 3))
        mass_array = np.zeros((self.molecule.natoms))
        for row in range(0, self.molecule.natoms):
            #adding 1/np.sqrt(amu) units to xyz coords
            x = element(self.molecule.atomtable[row][0])
            value = np.sqrt(x.atomic_weight)
            mass_array[row] = value
            mass_xyz[row] = np.array(self.molecule.atomtable[row][1:])*(1/value)   #mass weighted coordinates
            for column in range(0, len(self.molecule.atomtable)):    
                y = element(self.molecule.atomtable[column][0])
                z = x.atomic_weight*y.atomic_weight
                value = np.sqrt(z)**-1
                term = full_hessian[row][column]*value
                full_hessian[row][column] = term
                
        reshape_mass_hess = full_hessian.transpose(0, 2, 1, 3)
        x = reshape_mass_hess.reshape(reshape_mass_hess.shape[0]*reshape_mass_hess.shape[1],reshape_mass_hess.shape[2]*reshape_mass_hess.shape[3])
        e_values, modes = LA.eigh(x)
        #e_values, modes = LA.eigh(x)

        #unit conversion of freq from H/B**2 amu -> 1/s**2
        #factor = (4.3597482*10**-18)/(1.6603145*10**-27)/(1.0*10**-20)  #Angstrom to m
        factor = 1.8897259886**2*(4.3597482*10**-18)/(1.6603145*10**-27)/(1.0*10**-20) #Bohr to Angstrom
        freq = (np.sqrt(e_values*factor))/(2*np.pi*2.9979*10**10) #1/s^2 -> cm-1
        return freq, modes

    def global_apt(self):
        global_apt = np.zeros((self.molecule.natoms*3, 3))
        for frag in self.frags:
            #frag.build_apt()
            global_apt += frag.apt
        return global_apt


        


if __name__ == "__main__":
    #carbonylavo = Molecule()
    #carbonylavo.initalize_molecule('carbonylavo')
    #frag = Fragmentation(carbonylavo)
    #frag.do_fragmentation(frag_type='graphical', value=2)
    #frag.initalize_Frag_objects(theory='RHF', basis='sto-3g', qc_backend=Pyscf)
    #frag.energy_gradient(frag.moleculexyz)
    np.set_printoptions(suppress=True, precision=5)
    carbonylavo = Molecule.Molecule()
    carbonylavo.initalize_molecule('carbonylavo')
    frag = fragmentation.Fragmentation(carbonylavo)
    frag.do_fragmentation(frag_type='distance', value=1.8)
    frag.initalize_Frag_objects(theory='MP2', basis='sto-3g', qc_backend=Pyscf.Pyscf)
    #frag.energy_gradient(frag.moleculexyz)
   
    import ray
    start_time = time.time()
    ray.init()

    frags_id = ray.put(frag)    #future for Fragmentation instance, putting in object store

    @ray.remote
    def get_frag_stuff(f,_frags):
        f_current = _frags.frags[f]
        return f_current.qc_backend()
    result_ids = [get_frag_stuff.remote(fi, frags_id) for fi in range(len(frag.frags)) ]
    out = ray.get(result_ids)
    etot = 0
    gtot = 0
    htot = 0
    for o in out:
        etot += o[0]
        gtot += o[1]
        htot += o[2]
    total_time = time.time() - start_time 
    ray.shutdown()
    e, v = frag.mw_hessian(htot)
    print("e", e)
    print("modes", v)

    #start_fragtime = time.time()
    #frag.write_xyz('carbonylavo')
    #frag.energy_gradient(frag.moleculexyz)
    #end_time = time.time() - start_fragtime
    print("Final converged energy = ", etot)
    print("Final gradient = ", '\n', gtot)
    #print("Final hessian = ", '\n', htot)
    print("Hessian shape = ", htot.shape)
    #print(htot[0][1])
    #print(htot[1][0])


    ### run qc_params only if you want some fragments with different params
    #frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)
    
