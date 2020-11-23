import os
import sys
import numpy as np
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib
from input_file import *
#from sow import *
import shutil
import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import dill

path = sys.argv[1]
coords_name = sys.argv[2]

input_molecule = Molecule.Molecule(path)
input_molecule.initalize_molecule()
obj_list = []

if software == 'Pyscf':
    software = Pyscf.Pyscf
    
if mim_levels == 1:
    frag1 = fragmentation.Fragmentation(input_molecule)
    frag1.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag1.initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    os.path.abspath(os.curdir)
    os.chdir('to_run/')
    os.mkdir('frag1')
    obj_list.append(frag1)
    
if mim_levels == 2:
    #""" MIM high theory, small fragments"""
    frag1 = fragmentation.Fragmentation(input_molecule)
    print(type(frag1))
    print(frag1)
    frag1.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag1.initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    os.path.abspath(os.curdir)
    os.chdir('to_run/')
    os.mkdir('frag1')
    obj_list.append(frag1)
    
    #""" MIM low theory, small fragments"""
    frag2 = fragmentation.Fragmentation(input_molecule)
    frag2.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag2.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=-1)
    os.mkdir('frag2')
    obj_list.append(frag2)
    
    #""" MIM low theory, large fragments (iniffloate system)"""
    frag3 = fragmentation.Fragmentation(input_molecule)
    frag3.do_fragmentation(fragtype=frag_type, value=frag_deg_large)
    frag3.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    os.mkdir('frag3')
    obj_list.append(frag3)

levels = os.listdir()

def opt_fnc(newcoords):
    for atom in range(0, len(newcoords)): #makes newcoords = self.molecule.atomtable
        x = list(newcoords[atom])
        obj_list[0].molecule.atomtable[atom][1:] = x
    
    for j in range(0, len(obj_list)):       #update the other frag instances if MIM2 or higher level
        obj_list[j].molecule.atomtable = obj_list[0].molecule.atomtable
        os.chdir(levels[j])

        #remove old pickled objects and status
        frag_inst = os.listdir()
        for frag in frag_inst:
            shutil.rmtree(frag)
        
        #repickle fragment instances with new coords
        for i in range(0, len(obj_list[j].frags)):
            filename = "fragment" + str(i) + ".dill"
            outfile = open(filename, "wb")
            dill.dump(obj_list[j].frags[i], outfile)
            outfile.close()
        os.chdir('../')
    
    os.chdir('../')
    cmd = 'python batch.py %s'%(str(batch_size))
    print(cmd)
    os.system(cmd)
    etot = 0
    gtot = 0
    exit()
    etot = np.load(energy.npy)
    gtot = np.load(gradient.npy)
    return etot, gtot

obj_list[0].write_xyz(coords_name)
os.path.abspath(os.curdir)
optimizer = Berny(geomlib.readfile(os.path.abspath(coords_name)), debug=True)
x = 0
etot_opt = 0
grad_opt = 0
for geom in optimizer:
    x = x+1
    print("opt cycle:", x)
    solver = opt_fnc(geom.coords)
    optimizer.send(solver)
    etot_opt = solver[0]
    grad_opt = solver[1]
relaxed = geom
print("\n", "##########################", '\n', "#       Converged!       #", '\n', "##########################") 
print('\n', "Energy = ", etot_opt)
print('\n', "Converged_Gradient:", "\n", grad_opt)

