import numpy as np
from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad
from pyscf.grad.rhf import GradientsBasics
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib
#from berny.solvers import MopacSolver
#from pyscf.geomopt import berny_solver

#import autograd.numpy as auto
#import autograd.numpy as np
#from autograd import (grad, elementwise_grad, jacobian, value_and_grad,grad_and_aux, hessian_vector_product, hessian, multigrad, jacobian, vector_jacobian_product)
#from autograd import grad as a_grad
#from autograd import hessian as a_hess

def do_pyscf(input_xyz, theory, basis, hess=True):
    mol = gto.Mole()
    mol.atom = input_xyz
    mol.basis = basis
    mol.build()
    #m = int() 
    if theory == 'full':
        hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
        e, g = hf_scanner(mol)
        opt = optimize(scf.RHF(mol).kernel())
        return e, g

    if theory == 'RHF': #Restricted HF calc
        hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
        e, g = hf_scanner(mol)
        if hess == True:
            mf = mol.RHF().run()
            h = mf.Hessian().kernel()
        if hess == False:
            h = 0
        return e, g, h
    
    ##################################################################
    # NEED TO FIGURE OUT HOW TO PULL OUT HESSIAN FOR HIGHER THEORIES #
    ##################################################################

    if theory == 'MP2': #Perturbation second order calc
        mp2_scanner = mp.MP2(scf.RHF(mol)).nuc_grad_method().as_scanner()
        e, g = mp2_scanner(mol) 
        if hess == True:
            pass
        if hess == False:
            h = 0
        return e, g, h

    if theory == 'CISD':    #CI for singles and double excitations
        ci_scanner = ci.CISD(scf.RHF(mol)).nuc_grad_method().as_scanner()
        e, g = ci_scanner(mol)
        if hess == True:
            mf = ci.CISD(scf.RHF(mol)).run()
            h = mf.Hessian().kernel()
        else:
            h = 0
        return e, g, h

    if theory == 'CCSD':    #Couple Cluster for singles and doubles
        cc_scanner = cc.CCSD(scf.RHF(mol)).nuc_grad_method().as_scanner()
        e, g = cc_scanner(mol)
        if hess == True:
            mf = cc.CCSD(scf.RHF(mol)).run()
            h = mf.Hessian().kernel()
        else:
            h = 0
        return e, g, h

   
#mol.symmetry = 1
    #mol.charge = 1
    #mol.spin = 2   #This is 2S, difference between number of alpha and beta electrons
    #mol.verbose = 5    #this sets the print level globally 0-9
    #mol.output = 'path/to/my_log.txt'  #this writes the ouput messages to certain place
    #mol.max_memory = 1000 #MB  #defaul size can be defined withshell environment variable PYSCF_MAX_MEMORY
        #can also set memory from command line:
        #python example.py -o /path/to/my_log.txt -m 1000
    #mol.output = 'output_log'
    

