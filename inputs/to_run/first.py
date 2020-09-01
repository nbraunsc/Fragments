import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import os

largermol = Molecule.Molecule()
largermol.initalize_molecule('largermol')
frag = fragmentation.Fragmentation(largermol)
frag.do_fragmentation(frag_type='distance', value=1.8)
frag.initalize_Frag_objects(theory='RHF', basis='sto-3g', qc_backend=Pyscf.Pyscf, step_size=0.001)

#os.path.abspath(os.curdir)
#os.chdir('/to_run/')

#writing individual input files
x = 0
for i in frag.frags:
    inputxyz = i.build_xyz()
    x = x + 1
    name = "fragment" + str(x) + ".py"
    f = open(name, "w+")
    imports = ["import nicolefragment\n", "from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf\n", "import numpy as np\n", "import os\n", "from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf\n", "from pyscf.prop.freq import rhf\n", "from pyscf.prop.polarizability import rhf\n", "from pyscf.grad.rhf import GradientsBasics\n", " \n", "np.set_printoptions(suppress=True, precision=5)\n", " \n"]
    f.writelines(imports)
    build = ["mol = gto.Mole()\n", "mol.atom =", str(inputxyz), "\n", "mol.basis = 'sto-3g'\n", "mol.build()\n", "hf_scanner = scf.RHF(mol).apply(grad.RHF).as_scanner()\n", "e, g = hf_scanner(mol)\n", "mf = mol.RHF().run()\n", "h = mf.Hessian().kernel()\n", "print('energy' = e)\n", "print('gradient' = g)\n", "print('hessian' = h)"]
    f.writelines(build) 
    f.close()

files = os.listdir()
#os.chdir("../")

#writing individual pbs scripts for each file
for i in files:
    if i.startswith("fragment"):
        x = i.replace('.py', '')
        file_name = open("second_pbs.sh", "w")
        top = ["#PBS -l nodes=2:ppn=4\n", "#PBS -l mem=20GB\n", "#PBS -q nmayhall_lab\n", "#PBS -A qcvt_doe\n", "#PBS -W group_list=nmayhall_lab\n", " \n", "module purge\n", "module load gcc/5.2.0\n", "module load Anaconda/5.2.0\n", "$MKL_NUM_THREADS = 1\n", "cd $PBS_O_WORKDIR\n", "source activate pyconda\n", "cd ../../\n", "python -m pip install -e . \n", "cd $PBS_O_WORKDIR\n", "FILE=", x, "\n", "python $FILE.py >> $FILE.out\n", "exit;\n"]
        file_name.writelines(top)
        file_name.close()
        os.system("qsub second_pbs.sh")


#"""do_MIM1(deg, frag_type,  theory, basis, Molecule, opt=False, step=0.001)"""
#do_MIM1(3, 'distance', 'MP2', 'sto-3g', largermol, opt=False, step_size=0.001)        #uncomment to run MIM1

#"""do_MIM2(frag_type, frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule, opt=False)"""
#do_MIM2('distance', 1.3, 'MP2', 'ccpvdz', 1.8, 'RHF', 'ccpvdz', largermol, opt=False) #uncomment to run MIM2
