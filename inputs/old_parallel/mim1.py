import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import os

############## User defined paramters ###############
frag_type = 'distance'  #can be 'distance' or 'graphical'
frag_deg = 1.8  #smaller fragmentation level
basis = 'sto-3g'
theory = 'RHF'
software = Pyscf.Pyscf
stepsize = 0.001        #for second derivative by finite difference

largermol = Molecule.Molecule()         #largermol is what is replaced
largermol.initalize_molecule('largermol')
frag = fragmentation.Fragmentation(largermol)
##################################################

""" Setting up fragmentation for MIM1"""
frag.do_fragmentation(frag_type=str(frag_type), value=frag_deg)
frag.initalize_Frag_objects(theory=str(theory), basis=str(basis), qc_backend=software, step_size=stepsize)

#changing into the to_run directory
os.path.abspath(os.curdir)
os.chdir('to_run/')

#writing individual fragment input .py files to be run for RHF
x = 0
mim_coeff = 1
class_list = [frag]   #fragmentation instances
fraglist = ['frag']
for level in range(0, len(class_list)):
    os.mkdir(fraglist[level])
    os.chdir(fraglist[level])
    np.save(os.path.join('mim_co.npy'), mim_coeff)
    for i in class_list[level].frags:
        x = x + 1
        inputxyz = []
        inputxyz = i.build_xyz()
        len_prim = len(i.prims)
        num_la = len(i.notes)
        step_size = i.step_size
        basis_set = i.qc_class.basis
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
        info = ["len_prim =", str(len_prim), "\n"
        "num_la =", str(num_la), "\n"
        "step_size = ", str(step_size), "\n"]
        f.writelines(info)
        build = [
        "mol = gto.Mole()\n"
        "mol.atom = ", str(inputxyz), "\n", "mol.basis = '", basis_set, "'\n", "mol.build()\n", "mf = scf.RHF(mol)\n"]
        f.writelines(build) 
        pyscf = open('../../../nicolefragment/Pyscf.py', 'r')
        lines = pyscf.readlines()
    
        #writes scf info depending on theory given
        grad_line = str()
        if i.qc_class.theory == 'RHF': 
            egh = ["""e = mf.kernel()
g = mf.nuc_grad_method().kernel()
h = mf.Hessian().kernel()"""]
            grad_line = str('e = mf2.kernel()\n' + '            g = mf2.nuc_grad_method().kernel()')
        if i.qc_class.theory == 'MP2':
            egh = ["postmf = mp.MP2(mf).run()\n", 
            "e = mf.kernel() + postmf.kernel()[0]\n",
            "g = postmf.nuc_grad_method().kernel()\n",
            "h = 0\n"]
            grad_line = str('postmf2 = mp.MP2(mf2).run()\n' + '            e = mf2.kernel() + postmf2.kernel()[0]\n' + '            g = postmf2.nuc_grad_method().kernel()')
        if i.qc_class.theory == 'CISD':
            egh =  ["""postmf = ci.CISD(mf).run()
e = postmf.kernel()
g = postmf.nuc_grad_method().kernel()
h = 0"""]
            grad_line = str()
        #dipole = ["mfx = scf.RHF(mol).run()\n, dipole = mfx.dip_moment(mol)\n"]
        f.writelines(egh)
        #f.writelines(dipole)
        pyscf.close()
        num_hess = ["\n#If not analytical hess, not do numerical below\n",
"if type(h) is int:\n",
"    hess = np.zeros(((len(mol.atom))*3, (len(mol.atom))*3))\n",
"    i = -1\n",
"    for atom in range(0, len(mol.atom)):\n",
"        for xyz in range(0, 3):\n",
"            i = i+1\n",
"            mol.atom[atom][1][xyz] = mol.atom[atom][1][xyz]+step_size\n",
"            mol2 = gto.Mole()\n",
"            mol2.atom = mol.atom\n",
"            mol2.basis = mol.basis\n",
"            mol2.build()\n",
"            mf2 = scf.RHF(mol2)\n",
"            ", grad_line, "\n",
"            grad1 = g.flatten()\n",
"            mol.atom[atom][1][xyz] = mol.atom[atom][1][xyz]-2*step_size\n",
"            mol2 = gto.Mole()\n",
"            mol2.atom = mol.atom\n",
"            mol2.basis = mol.basis\n",
"            mol2.build()\n",
"            ", grad_line, "\n",
"            grad2 = g.flatten()\n",
"            mol.atom[atom][1][xyz] = mol.atom[atom][1][xyz]+step_size\n",
"            vec = (grad1 - grad2)/(4*step_size)\n",
"            hess[i] = vec\n",
"            hess[:,i] = vec\n",
"    h = hess.reshape(len_prim+num_la, 3, len_prim+num_la, 3)\n",
"    h = h.transpose(0, 2, 1, 3)\n"]
        f.writelines(num_hess)
        scf_info = ["print('energy no coeff =', e)\n", "print('gradient =', g)\n", "print('hessian =', h)\n", "np.save(os.path.join('", str(x),"', '", name, "e.npy'), e)\n", "np.save(os.path.join('", str(x),"', '", name, "g.npy'), g)\n", "np.save(os.path.join('", str(x), "', '", name, "h.npy'), h)\n"]
        f.writelines(scf_info) 
        f.close()
    os.chdir('../')

#writing individual pbs scripts for each file
for level in fraglist:
    os.chdir(level)
    files = os.listdir()
    for i in files:
        if i.startswith("fragment"):
            x = i.replace('.py', '')
            filename = x + ".sh"
            file_name = open(filename, "w")
            top = ["#PBS -l nodes=2:ppn=4\n", "#PBS -l mem=20GB\n", "#PBS -q nmayhall_lab\n", "#PBS -A qcvt_doe\n", "#PBS -W group_list=nmayhall_lab\n", " \n", "module purge\n", "module load gcc/5.2.0\n", "module load Anaconda/5.2.0\n", "$MKL_NUM_THREADS = 1\n", "cd $PBS_O_WORKDIR\n", "source activate pyconda\n", "cd ../../\n", "python -m pip install -e . \n", "cd $PBS_O_WORKDIR\n", "FILE=", x, "\n", "python $FILE.py >> $FILE.log\n", "exit;\n"]
            file_name.writelines(top)
            file_name.close()
