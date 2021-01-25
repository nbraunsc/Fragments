import numpy as np
from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf, ao2mo
from pyscf.prop.freq import rhf
from pyscf.prop.polarizability import rhf
from pyscf.grad.rhf import GradientsBasics
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib

class Pyscf():
    """ Pyscf quantum chemistry backend class
    
    An instance of this class is passed into the Fragment class
    """

    def __init__(self, theory=None, basis=None, spin=None, tol=None, active_space=None, nelec=None,  nelec_alpha=None, nelec_beta=None, max_memory=None):
        self.theory = theory
        self.basis = basis
        self.spin = spin
        self.tol = tol
        self.active_space = active_space    #number of orbitals in active space
        self.nelec = nelec  #number of electrons in the active space
        self.nelec_alpha = nelec_alpha
        self.nelec_beta = nelec_beta
        self.max_memory = max_memory
        #self.input_xyz = []

    def energy_gradient(self, input_xyz):
        mol = gto.Mole()
        mol.atom = input_xyz
        mol.basis = self.basis
        mol.symmetry = True
        mol.build()
        mf = scf.RHF(mol).run()
    
        if self.theory == 'RHF': #Restricted HF calc
            e = mf.kernel()
            g = mf.nuc_grad_method().kernel()
            h = mf.Hessian().kernel()
            return e, g, h
    
        if self.theory == 'MP2': #Perturbation second order calc
            postmf = mp.MP2(mf).run()
            e = mf.kernel() + postmf.kernel()[0]
            g2 = postmf.nuc_grad_method()
            g = g2.kernel()
            h = 0
            return e, g, h
    
        if self.theory == 'CISD':    #CI for singles and double excitations
            postmf = ci.CISD(mf).run()
            e = postmf.kernel()
            g2 = postmf.nuc_grad_method()
            g = g2.kernel()
            h = 0
            return e, g, h
    
        if self.theory == 'CCSD':    #Couple Cluster for singles and doubles
            postmf = cc.CCSD(mf).run()
            e = postmf.kernel()
            g2 = postmf.nuc_grad_method()
            g = g2.kernel()
            h = 0
            return e, g, h

        if self.theory == 'CASSCF':
            postmf = mcscf.CASSCF(mf, self.active_space, self.nelec).run()
            e = postmf.kernel()
            g2 = postmf.nuc_grad_method()
            g = g2.kernel()
            h = 0
            return e, g, h

        if self.theory == 'CASCI':
            h = 0
            return e, g, h
    
    def apply_field(self, E, input_xyz):
        """ This will apply an electric field in a specific direction for pyscf. gives E vector
        to make a new hcore. Used when building the derivative of gradient w.r.t electric field to
        compute the IR intensities.
    
        Parameters
        ----------
        E : np array
            This is a 1D array of an x, y, z.  Put magintude of wanted E field in the position of the.  Right now it is only set up for RHF.

            direction wanted.
        input_xyz : list
            Coordinates for molecule
        Returns
        -------
        mos : ndarray?
            This are the new mos in the core hamiltonian used for another SCF calculation.
            Dont really need these
    
        """
        e=0
        g2=0
        dipole1=0
        dm_init_guess = [None]
        mol1 = gto.Mole()
        mol1.atom = input_xyz
        mol1.basis = self.basis
        #mol1.symmetry = True   #if set to True x and y grads are zero in APT
        mol1.unit = 'Angstrom'
        mol1.build()
        #mfx = scf.RHF(mol1).run()
        #dip_unit = 'DEBYE'
        #dip_unit2 = 'A.U.'
        #dipole1 = mfx.dip_moment(mol1, unit=dip_unit)
        #dipole1 = mfx.dip_moment(mol1, unit=dip_unit2)
        
        mol1.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
        h =(mol1.intor('cint1e_kin_sph') + mol1.intor('cint1e_nuc_sph')
          + np.einsum('x,xij->ij', E, mol1.intor('cint1e_r_sph', comp=3)))
        mfx = scf.RHF(mol1)
        mfx.get_hcore = lambda *args: h
        mfx.scf(dm_init_guess[0])
        dm_init_guess[0] = mfx.make_rdm1()
        mol1.incore_anyway = True    #needed for post HF calculations to make sure custom H is used
        
        e = mfx.kernel()
        g2 = mfx.nuc_grad_method().kernel()    #only gradient for RHF right now
        print("################\n")
        print("######### PYSCF PRINTS ##########")
        print(mol1.atom_coords())
        print(E)
        #print("energy:", e)
        #print("dipole moment:", dipole1, dip_unit)
        #print("dipole moment:", dipole2, dip_unit2)
        print("################\n")
        return e, g2, dipole1

    
    def get_dipole(self, coords_new):
        """ This only for RHF and is used when building the APT's from numerical diff w.r.t 
        atomic coordinates.
        """

        mol2 = gto.Mole()
        mol2.atom = coords_new
        mol2.basis = self.basis
        mol2.symmetry = True
        mol2.unit = 'Angstrom'
        mol2.build()
        mfx = scf.RHF(mol2).run()
        dipole_hf = mfx.dip_moment(mol2, unit='DEBYE')
        return dipole_hf
    




    


