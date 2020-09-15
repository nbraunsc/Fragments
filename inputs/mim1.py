import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import os

#starting fragmentation code
largermol = Molecule.Molecule()
largermol.initalize_molecule('largermol')
frag = fragmentation.Fragmentation(largermol)
frag.do_fragmentation(frag_type='distance', value=1.8)
frag.initalize_Frag_objects(theory='RHF', basis='sto-3g', qc_backend=Pyscf.Pyscf, step_size=0.001)

#changing into the to_run directory
os.path.abspath(os.curdir)
os.chdir('to_run/')

#writing individual fragment input .py files to be run for RHF
x = 0
for i in frag.frags:
    x = x + 1
    inputxyz = []
    inputxyz = i.build_xyz()
    #makes a directory for each fragment
    os.mkdir(str(x))
    #computes the jacobians and saves as a .npy file in dir
    g_jacob = i.build_jacobian_Grad()
    h_jacob = i.build_jacobian_Hess()
    coeff = i.coeff
    np.save(os.path.join(str(x), 'jacob_grad'+ str(x) + '.npy'), g_jacob)
    np.save(os.path.join(str(x), 'jacob_hess'+ str(x) + '.npy'), h_jacob)
    np.save(os.path.join(str(x), 'coeff'+str(x) + '.npy'), coeff)
    name = "fragment" + str(x) + ".py"
    #start of writing individual fragment files
    f = open(name, "w+")
    imports = ["""import numpy as np\n
import os\n
from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf\n
from pyscf.prop.freq import rhf\n
from pyscf.prop.polarizability import rhf\n
from pyscf.grad.rhf import GradientsBasics\n
np.set_printoptions(suppress=True, precision=5)\n"""]
    f.writelines(imports)
    build = [
    "mol = gto.Mole()\n"
    "mol.atom = ", str(inputxyz), "\n", "mol.basis = 'sto-3g'\n", "mol.build()\n"]
    f.writelines(build) 
    pyscf = open('../../nicolefragment/Pyscf.py', 'r')
    lines = pyscf.readlines()
    
    #writes scf info depending on theory given
    emp = []
    if i.qc_class.theory == 'RHF': 
        indices = [38, 39, 40, 41]
        for index in indices:
            emp.append(lines[index].lstrip())
    if i.qc_class.theory == 'MP2':
        indices = [45, 46, 47]
        for index in indices:
            emp.append(lines[index].lstrip())
    if i.qc_class.theory == 'CISD':
        indices = [51, 52, 53]
        for index in indices:
            emp.append(lines[index].lstrip())
    #dipole = ["mfx = scf.RHF(mol).run()\n, dipole = mfx.dip_moment(mol)\n"]
    f.writelines(x for x in emp)
    #f.writelines(dipole)
    pyscf.close()
    scf_info = ["print('energy no coeff =', e)\n", "print('gradient =', g)\n", "print('hessian =', h)\n", "np.save(os.path.join('", str(x),"', '", name, "e.npy'), e)\n", "np.save(os.path.join('", str(x),"', '", name, "g.npy'), g)\n", "np.save(os.path.join('", str(x), "', '", name, "h.npy'), h)\n"]
    f.writelines(scf_info) 
    f.close()

files = os.listdir()
#writing individual pbs scripts for each file
for i in files:
    if i.startswith("fragment"):
        x = i.replace('.py', '')
        filename = x + ".sh"
        file_name = open(filename, "w")
        top = ["#PBS -l nodes=2:ppn=4\n", "#PBS -l mem=20GB\n", "#PBS -q nmayhall_lab\n", "#PBS -A qcvt_doe\n", "#PBS -W group_list=nmayhall_lab\n", " \n", "module purge\n", "module load gcc/5.2.0\n", "module load Anaconda/5.2.0\n", "$MKL_NUM_THREADS = 1\n", "cd $PBS_O_WORKDIR\n", "source activate pyconda\n", "cd ../../\n", "python -m pip install -e . \n", "cd $PBS_O_WORKDIR\n", "FILE=", x, "\n", "python $FILE.py >> $FILE.log\n", "exit;\n"]
        file_name.writelines(top)
        file_name.close()
