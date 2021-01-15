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
import glob
import threading
import time

path = sys.argv[1]
coords_name = path.replace(".cml", ".xyz")

input_molecule = Molecule.Molecule(path)
input_molecule.initalize_molecule()
obj_list = []
name_list = []

if software == 'Pyscf':
    software = Pyscf.Pyscf
    
if mim_levels == 1:
    frag1 = fragmentation.Fragmentation(input_molecule)
    frag1.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag1.initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    os.path.abspath(os.curdir)
    os.chdir('to_run/')
    level_list = os.listdir()
    for level in level_list:
        shutil.rmtree(level)
    os.mkdir('frag1')
    obj_list.append(frag1)
    name_list.append('frag1')
    
if mim_levels == 2:
    #""" MIM high theory, small fragments"""
    frag1 = fragmentation.Fragmentation(input_molecule)
    frag1.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag1.initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    os.path.abspath(os.curdir)
    os.chdir('to_run/')
    level_list = os.listdir()
    for level in level_list:
        shutil.rmtree(level)
    os.mkdir('frag1')
    obj_list.append(frag1)
    name_list.append('frag1')
    
    #""" MIM low theory, small fragments"""
    frag2 = fragmentation.Fragmentation(input_molecule)
    frag2.do_fragmentation(fragtype=frag_type, value=frag_deg)
    frag2.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=-1)
    os.mkdir('frag2')
    obj_list.append(frag2)
    name_list.append('frag2')
    
    #""" MIM low theory, large fragments (iniffloate system)"""
    frag3 = fragmentation.Fragmentation(input_molecule)
    frag3.do_fragmentation(fragtype=frag_type, value=frag_deg_large)
    frag3.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    os.mkdir('frag3')
    obj_list.append(frag3)
    name_list.append('frag3')

os.chdir('../')

def opt_fnc(newcoords):
    os.chdir('to_run/')
    for atom in range(0, len(newcoords)): #makes newcoords = self.molecule.atomtable
        x = list(newcoords[atom])
        obj_list[0].molecule.atomtable[atom][1:] = x
    
    for j in range(0, len(obj_list)):       #update the other frag instances if MIM2 or higher level
        obj_list[j].molecule.atomtable = obj_list[0].molecule.atomtable
        os.chdir(name_list[j])
        files = os.listdir()
        print("files in", name_list[j], files)

        #remove old pickled objects and status
        for filename in files:
            os.remove(filename)
        print("empty dir:", os.listdir())

        #repickle fragment instances with new coords
        for i in range(0, len(obj_list[j].frags)):
            filename = "fragment" + str(i) + ".dill"
            outfile = open(filename, "wb")
            dill.dump(obj_list[j].frags[i], outfile)
            outfile.close()
        os.chdir('../')
    os.chdir('../')
    
    #deleting old energy and gradient numpy objects between itertions
    print("removing energy.npy and grad.npy")
    npy_list = glob.glob('*.npy')
    for thing in npy_list:
        os.remove(thing)

    cmd = 'python batch.py %s pbs.sh'%(str(batch_size))
    #opt_cmd = 'qsub -N checker geom_opt.sh'
    print(cmd)
    #print(opt_cmd)
    os.system(cmd)
    #os.system(opt_cmd)
    etot = 0
    gtot = 0

    #pauses python function until all batches are done running and global etot and gtot calculated
    #while len(glob.glob("*.npy")) < 2:
    #    print("sleeping in while loop")
    #    time.sleep(30)
    #    print("done sleeping")
    #    pass

    #load in the etot and gtot
    etot = np.load('energy.npy')
    gtot = np.load('gradient.npy')
    print("energy from .npy = ", etot)
    return etot, gtot


obj_list[0].write_xyz(coords_name)
os.path.abspath(os.curdir)
optimizer = Berny(geomlib.readfile(os.path.abspath(coords_name)), debug=True)
x = 0
etot_opt = 0
grad_opt = 0
for geom in optimizer:
    x = x+1
    print("\n opt cycle:", x, "\n")
    solver = opt_fnc(geom.coords)
    optimizer.send(solver)
    etot_opt = solver[0]
    grad_opt = solver[1]
relaxed = geom

print("\n", "##########################", '\n', "#       Converged!       #", '\n', "##########################") 
print('\n', "Energy = ", etot_opt)
print('\n', "Converged_Gradient:", "\n", grad_opt)

print("\n", relaxed.coords, "\n")
os.chdir('to_run/')

#updating coords with optimized geometry for hessian
for atom in range(0, len(relaxed.coords)): #makes newcoords = self.molecule.atomtable
    x = list(relaxed.coords[atom])
    obj_list[0].molecule.atomtable[atom][1:] = x

for j in range(0, len(obj_list)):       #update the other frag instances if MIM2 or higher level
    obj_list[j].molecule.atomtable = obj_list[0].molecule.atomtable
    os.chdir(name_list[j])

    #remove old pickled objects and status
    for filename in os.listdir():
        os.remove(filename)

    #repickle fragment instances with new coords
    for i in range(0, len(obj_list[j].frags)):
        filename = "fragment" + str(i) + ".dill"
        outfile = open(filename, "wb")
        dill.dump(obj_list[j].frags[i], outfile)
        outfile.close()
    os.chdir('../')
os.chdir('../')

#Running hessian and apt at optimized geometry
cmd = 'python batch.py %s hess_apt.sh'%(str(batch_size))
print(cmd)
os.system(cmd)




