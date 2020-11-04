import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
#from dummie_input import *

import numpy as np
import os
import sys
import pickle

input_file = sys.argv[1]

from input_file import *

coords = sys.argv[2]
path = os.getcwd() + "/" + coords

if software == 'Pyscf':
    software = Pyscf.Pyscf

#if software == "Molcas" or "OpenMolcas":
#    software = Molcas.Molcas

if mim_levels == None:
    raise Exception('MIM level not defined')
    
if not frag_type:
    raise Exception('Fragmentation type not defined')
    
if frag_deg == None:
    raise Exception('Fragmentation distance cutoff not defined')
    
if frag_deg_large == None:
    raise Exception('Second fragmentation distance cutoff not defined')
    
if not basis_set:
    raise Exception('Basis set not defined')
    
if mim_levels == 2 and not low_theory: 
    raise Exception('Low level of theory not defined')

if not high_theory:
    raise Exception('High level of theory not defined')
    
if not software:
    raise Exception('Software for calculations not defined')
    
if stepsize == None:
    raise Exception('Step size for finite difference not defined')


input_molecule = Molecule.Molecule(path)
input_molecule.initalize_molecule()

if mim_levels == 1:
    frag = fragmentation.Fragmentation(input_molecule)
    frag.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag.initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    os.path.abspath(os.curdir)
    os.chdir('to_run/')
    os.mkdir('frag1')
    os.chdir('frag1')
    for i in range(0, len(frag.frags)):
        filename = "fragment" + str(i)
        outfile = open(filename, "wb")
        pickle.dump(frag.frags[i], outfile)
        outfile.close()
    
if mim_levels == 2:
    #""" MIM high theory, small fragments"""
    frag1 = fragmentation.Fragmentation(input_molecule)
    frag1.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag1.initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    os.path.abspath(os.curdir)
    os.chdir('to_run/')
    os.mkdir('frag1')
    os.chdir('frag1')
    for i in range(0, len(frag1.frags)):
        filename = "fragment" + str(i)
        outfile = open(filename, "wb")
        pickle.dump(frag1.frags[i], outfile)
        outfile.close()
    os.chdir('../')
    
    #""" MIM low theory, small fragments"""
    frag2 = fragmentation.Fragmentation(input_molecule)
    frag2.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag2.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=-1)
    os.mkdir('frag2')
    os.chdir('frag2')
    for i in range(0, len(frag2.frags)):
        filename = "fragment" + str(i)
        outfile = open(filename, "wb")
        pickle.dump(frag2.frags[i], outfile)
        outfile.close()
    os.chdir('../')
    
    #""" MIM low theory, large fragments (iniffloate system)"""
    frag3 = fragmentation.Fragmentation(input_molecule)
    frag3.do_fragmentation(fragtype=frag_type, value=frag_deg_large)
    frag3.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    os.mkdir('frag3')
    os.chdir('frag3')
    for i in range(0, len(frag3.frags)):
        filename = "fragment" + str(i)
        outfile = open(filename, "wb")
        pickle.dump(frag3.frags[i], outfile)
        outfile.close()
    os.chdir('../')
