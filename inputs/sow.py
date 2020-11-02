import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import os
import sys

input_file = sys.argv[1]
f = open(input_file, 'r')
lines = f.readlines()
for line in lines:
    if line.startswith("molecule_name"):
        x = line.split("=", 1)[1]
        molecule_name = x.rstrip()
        if not x:
            raise Exception('No molecule defined')

    if line.startswith("mim_type"):
        x = line.split("=", 1)[1]
        mim_type = x.strip()
        if not x:
            raise Exception('MIM level not defined')
    
    if line.startswith("fragtype"):
        x = line.split("=", 1)[1]
        frag_type = x.strip()
        if not x:
            raise Exception('Fragmentation type not defined')
    
    if line.startswith("frag_deg"):
        x = line.split("=", 1)[1].replace("'", "")
        frag_deg = float(x)
        if not x:
            raise Exception('Fragmentation distance cutoff not defined')
    
    if line.startswith("large_deg2"):
        x = line.split("=", 1)[1].replace("'", "")
        frag_deg2 = float(x)
        if not x:
            raise exception('Second fragmentation distance cutoff not defined')
    
    if line.startswith("basis_set"):
        x = line.split("=", 1)[1]
        basis_set = x.strip()
        if not x:
            raise Exception('Basis set not defined')
    
    if line.startswith("low_theory"):
        x = line.split("=", 1)[1]
        low_theory = x.strip()
        if not x:
            raise Exception('Low level of theory not defined')

    if line.startswith("high_theory"):
        x = line.split("=", 1)[1]
        high_theory = x.strip()
        if not x:
            raise Exception('High level of theory not defined')
    
    if line.startswith("software"):
        x = line.split("=", 1)[1].replace("'", "")
        software = x.strip()
        if not x:
            raise Exception('Software for calculations not defined')
    
    if line.startswith("stepsize"):
        x = line.split("=", 1)[1]
        stepsize = float(x)
        if not x:
            raise Exception('Step size for finite difference not defined')


input_molecule = Molecule.Molecule(molecule_name)
input_molecule.initalize_molecule()
frag = fragmentation.Fragmentation(input_molecule)
frag.do_fragmentation(fragtype=frag_type, value=frag_deg)
frag.initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize)
print(frag.frags)

print(input_molecule, " = Molecule.Molecule(", molecule_name, ")")
print(input_molecule, ".initalize_molecule()")
print("frag = fragmentation.Fragmentation(", input_molecule, ")")
print("frag.do_fragmentation(fragtype=", frag_type, ",value=", frag_deg, ")")
print("frag.initalize_Frag_objects(theory=", high_theory, "basis=", basis_set, ",qc_backend=", software, ",step_size=", stepsize, ")")


if mim_type.startswith('mim1'):
    frag = fragmentation.Fragmentation(input_molecule)
    frag.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag.initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize)
    for i in range(0, len(frag.frags)):
        filename = "fragment" + str(i)
        outfile = open(filename, "wb")
        pickle.dump(frag.frags[i], outfile)
        outfile.close()


if mim_type.startswith('mim2'):
    print('hi')
    #""" MIM high theory, small fragments"""
    frag1 = fragmentation.Fragmentation(input_molecule)
    frag1.do_fragmentation(frag_type=fragtype, value=frag_deg)
    frag1.initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize)
    print(frag1.frags)
    
    #""" MIM low theory, small fragments"""
    frag2 = fragmentation.Fragmentation(input_molecule)
    frag2.do_fragmentation(frag_type=fragtype, value=frag_deg)
    frag2.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize)
    
    #""" MIM low theory, large fragments (iniffloate system)"""
    frag3 = fragmentation.Fragmentation(input_molecule)
    frag3.do_fragmentation(frag_type=fragtype, value=frag_deg2)
    frag3.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize)
