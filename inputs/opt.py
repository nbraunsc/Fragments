import os
import sys
import numpy as np
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib
from input_file import *
from sow import *

objs = sys.argv[1]

def opt_fnc(newcoords):
    for atom in range(0, len(newcoords)): #makes newcoords = self.molecule.atomtable
        x = list(newcoords[atom])
        objs[0].molecule.atomtable[atom][1:] = x
    
    for i in range(0, len(objs)):       #update the other frag instances if MIM2 or higher level
        objs[i].molecule.atomtable = objs[0].molecule.atomtable
        objs[i].initalize_Frag_objects(theory=high_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)
    frag2.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=-1)
    frag3.initalize_Frag_objects(theory=low_theory, basis=basis_set, qc_backend=software, step_size=stepsize, local_coeff=1)

    
    MIM2_energy = 0
    MIM2_grad = np.zeros((frag1.molecule.natoms, 3))
    
    etot1, gtot1, htot1, apt1 = global_props(frag1, step=0.001)
    etot2, gtot2, htot2, apt2 = global_props(frag2, step=0.001)
    etot3, gtot3, htot3, apt3 = global_props(frag3, step=0.001)
    MIM2_energy = etot1 - etot2 + etot3
    MIM2_grad = gtot1 - gtot2 + gtot3
    return MIM2_energy, MIM2_grad

objs[0].write_xyz(input_molecule)
os.path.abspath(os.curdir)
#os.chdir('../inputs/')
optimizer = Berny(geomlib.readfile(os.path.abspath(os.curdir) + '/' + str(Molecule) + '.xyz'), debug=True)
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

cmd = 'python batch.py %s'%(batch_size)
