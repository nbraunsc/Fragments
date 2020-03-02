import numpy as np
from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad
from pyscf.grad.rhf import GradientsBasics
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib
#from berny.solvers import MopacSolver
#from pyscf.geomopt import berny_solver


def do_pyscf(input_xyz, theory, basis):
    mol = gto.Mole()
    mol.atom = input_xyz
    mol.basis = basis
    mol.build()
    #m = int() 
    if theory == 'RHF': #Restricted HF calc
        hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()
        e, g = hf_scanner(mol)
        return e, g
    
    if theory == 'MP2': #Perturbation second order calc
        mp2_scanner = mp.MP2(scf.RHF(mol)).nuc_grad_method().as_scanner()
        e, g = mp2_scanner(mol) 
        
        #print('--------------- RHF Hessian ------------------','\n', h, '\n', '----------------------------------------------')
        #print(mol_eq.atom_coords())
        return e, g

    if theory == 'CISD':    #CI for singles and double excitations
        ci_scanner = ci.CISD(scf.RHF(mol)).nuc_grad_method().as_scanner()
        e, g = ci_scanner(mol)
        return e, g

    if theory == 'CCSD':    #Couple Cluster for singles and doubles
        cc_scanner = cc.CCSD(scf.RHF(mol)).nuc_grad_method().as_scanner()
        e, g = cc_scanner(mol)
        return e, g

    
   
#mol.symmetry = 1
    #mol.charge = 1
    #mol.spin = 2   #This is 2S, difference between number of alpha and beta electrons
    #mol.verbose = 5    #this sets the print level globally 0-9
    #mol.output = 'path/to/my_log.txt'  #this writes the ouput messages to certain place
    #mol.max_memory = 1000 #MB  #defaul size can be defined withshell environment variable PYSCF_MAX_MEMORY
        #can also set memory from command line:
        #python example.py -o /path/to/my_log.txt -m 1000
    #mol.output = 'output_log'
    

