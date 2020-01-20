import numpy as np
from pyscf import gto, scf, hessian, mp, lib
from pyscf.grad.rhf import GradientsBasics
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib
from berny.solvers import MopacSolver
from pyscf.geomopt import berny_solver


def do_pyscf(input_xyz, theory, basis):
    mol = gto.Mole()
    mol.atom = input_xyz
    mol.basis = basis
    mol.build()
    m = int() 
    if theory == 'RHF': #Restricted HF calc
        m = scf.RHF(mol)
        energy = m.kernel()
        g = m.nuc_grad_method()
        grad = g.kernel()
    
    if theory == 'MP2': #Perturbation second order calc
        mp2_scanner = mp.MP2(scf.RHF(mol)).as_scanner()
        energy = mp2_scanner(mol)
        g = mp2_scanner.nuc_grad_method()
        grad = g.kernel()
        
        #h = m.Hessian().kernel()
        #print('--------------- RHF Hessian ------------------','\n', h, '\n', '----------------------------------------------')
        #mol_eq = optimize(m)
        #print(mol_eq.atom_coords())
        #print('\n', mol.atom)
    
    return energy, grad

def test_fn(etot, grad):
    e_total = etot
    grad_total = grad
    return e_total, grad_total

def do_geomopt(molecule, etot, grad, basis):
    mol = gto.Mole()
    mol.atom = molecule
    mol.basis = basis
    mol.build
    fake_method = berny_solver.as_pyscf_method(mol, test_fn(etot, grad))
    new_mol = berny_solver.optimize(fake_method)

    
   
#mol.atom =  '/home/nbraunsc/Documents/Projects/MIM/myoutfile.txt'   #different atoms are seperated by ; or a line break
#mol.symmetry = 1
    #mol.charge = 1
    #mol.spin = 2   #This is 2S, difference between number of alpha and beta electrons
    #mol.verbose = 5    #this sets the print level globally 0-9
    #mol.output = 'path/to/my_log.txt'  #this writes the ouput messages to certain place
    #mol.max_memory = 1000 #MB  #defaul size can be defined withshell environment variable PYSCF_MAX_MEMORY
        #can also set memory from command line:
        #python example.py -o /path/to/my_log.txt -m 1000
    #mol.output = 'output_log'
    

