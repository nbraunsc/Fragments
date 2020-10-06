import numpy as np
from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf
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
        mol.unit = 'Angstrom'
        mol.build()
        if self.theory == 'full':
            hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
            e, g = hf_scanner(mol)
            opt = optimize(scf.RHF(mol).kernel())
            return e, g
    
        if self.theory == 'RHF': #Restricted HF calc
            hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
            e, g = hf_scanner(mol)
            mf = mol.RHF().run()
            h = mf.Hessian().kernel()
            return e, g, h
    
        if self.theory == 'MP2': #Perturbation second order calc
            mp2_scanner = mp.MP2(scf.RHF(mol)).nuc_grad_method().as_scanner()
            e, g = mp2_scanner(mol) 
            h = 0
            return e, g, h
    
        if self.theory == 'CISD':    #CI for singles and double excitations
            ci_scanner = ci.CISD(scf.RHF(mol)).nuc_grad_method().as_scanner()
            e, g = ci_scanner(mol)
            h = 0
            return e, g, h
    
        if self.theory == 'CCSD':    #Couple Cluster for singles and doubles
            cc_scanner = cc.CCSD(scf.RHF(mol)).nuc_grad_method().as_scanner()
            e, g = cc_scanner(mol)
            h = 0
            return e, g, h

        if self.theory == 'CASSCF':
            
            mc_grad_scanner = mcscf.CASSCF(scf.RHF(mol), self.active_space, self.nelec).nuc_grad_method().as_scanner()
            e, g = mc_grad_scanner(mol, spin=self.spin)
            h = 0
            return e, g, h

        if self.theory == 'CASCI':
            mc_grad_scanner = mcscf.CASCI(scf.RHF(mol), self.active_space, self.nelec).nuc_grad_method().as_scanner()
            e, g = mc_grad_scanner(mol, spin=self.spin)
            h = 0
            return e, g, h
    
    def get_dipole(self, coords_new):
        mol2 = gto.Mole()
        mol2.atom = coords_new
        mol2.basis = self.basis
        mol2.unit = 'Angstrom'
        mol2.build()
        mfx = scf.RHF(mol2).run()
        dipole1 = mfx.dip_moment(mol2)
        return dipole1
    




    


