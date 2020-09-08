import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import os

#starting fragmentation code
adamantane = Molecule.Molecule()
adamantane.initalize_molecule('adamantane')
frag = fragmentation.Fragmentation(adamantane)
frag.do_fragmentation(frag_type='distance', value=1.3)
frag.initalize_Frag_objects(theory='RHF', basis='sto-3g', qc_backend=Pyscf.Pyscf, step_size=0.001)

#changing into the to_run directory
os.path.abspath(os.curdir)
os.chdir('to_run/')

#writing individual fragment input .py files to be run for RHF
x = 0
for i in frag.frags:
    x = x + 1
    inputxyz = i.build_xyz()
    os.mkdir(str(x))
    g_jacob = i.build_jacobian_Grad()
    h_jacob = i.build_jacobian_Hess()
    coeff = i.coeff
    np.save(os.path.join(str(x), 'jacob_grad'+ str(x) + '.npy'), g_jacob)
    np.save(os.path.join(str(x), 'jacob_hess'+ str(x) + '.npy'), h_jacob)
    np.save(os.path.join(str(x), 'coeff'+str(x) + '.npy'), coeff)
    name = "fragment" + str(x) + ".py"
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
    if i.qc_class.theory == 'RHF': 
        scf_info = ["hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()\n", "e, g = hf_scanner(mol)\n", "mf = mol.RHF().run()\n", "h = mf.Hessian().kernel()\n", "print('energy no coeff =', e)\n", "print('gradient =', g)\n", "print('hessian =', h)\n", "np.save(os.path.join('", str(x),"', '", name, "e.npy'), e)\n", "np.save(os.path.join('", str(x),"', '", name, "g.npy'), g)\n", "np.save(os.path.join('", str(x), "', '", name, "h.npy'), h)\n"]
        f.writelines(scf_info) 
    if i.qc_class.theory == 'MP2':
        scf_info = 
        ["""mp2_scanner = mp.MP2(scf.RHF(mol)).nuc_grad_method().as_scanner()\n
e, g = mp2_scanner(mol)\n 
h = 0\n"""]
        f.writelines(scf_info)
    f.close()

files = os.listdir()
#os.chdir("../")

#writing individual pbs scripts for each file
#could just do this once and then just edit the filename for each fragment input
for i in files:
    if i.startswith("fragment"):
        x = i.replace('.py', '')
        filename = x + ".sh"
        file_name = open(filename, "w")
        top = ["#PBS -l nodes=2:ppn=4\n", "#PBS -l mem=20GB\n", "#PBS -q nmayhall_lab\n", "#PBS -A qcvt_doe\n", "#PBS -W group_list=nmayhall_lab\n", " \n", "module purge\n", "module load gcc/5.2.0\n", "module load Anaconda/5.2.0\n", "$MKL_NUM_THREADS = 1\n", "cd $PBS_O_WORKDIR\n", "source activate pyconda\n", "cd ../../\n", "python -m pip install -e . \n", "cd $PBS_O_WORKDIR\n", "FILE=", x, "\n", "python $FILE.py >> $FILE.out\n", "exit;\n"]
        file_name.writelines(top)
        file_name.close()


#import nicolefragment\n 
#from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf\n
