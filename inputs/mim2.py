import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import os

############## User defined paramters ###############
frag_type = 'distance'  #can be 'distance' or 'graphical'
frag_deg = 1.8  #smaller fragmentation level
infinite_deg = 3        #larger fragmentation level or infinity
basis = 'sto-3g'
low_theory = 'RHF'
high_theory = 'MP2'
software = Pyscf.Pyscf
stepsize = 0.001        #for second derivative by finite difference

largermol = Molecule.Molecule()         #largermol is what is replaced
largermol.initalize_molecule('largermol')
frag1 = fragmentation.Fragmentation(largermol)
frag2 = fragmentation.Fragmentation(largermol)
frag3 = fragmentation.Fragmentation(largermol)
#####################################################

""" MIM high theory, small fragments"""
frag1.do_fragmentation(frag_type=str(frag_type), value=frag_deg)
frag1.initalize_Frag_objects(theory=str(high_theory), basis=str(basis), qc_backend=software, step_size=stepsize)
#frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)

""" MIM low theory, small fragments"""
frag2.do_fragmentation(frag_type=str(frag_type), value=frag_deg)
frag2.initalize_Frag_objects(theory=str(low_theory), basis=str(basis), qc_backend=software, step_size=stepsize)
#frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)

""" MIM low theory, large fragments (inifinte system)"""
frag3.do_fragmentation(frag_type=str(frag_type), value=infinite_deg)
frag3.initalize_Frag_objects(theory=str(low_theory), basis=str(basis), qc_backend=software, step_size=stepsize)
#frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)

#changing into the to_run directory
os.path.abspath(os.curdir)
os.chdir('to_run/')

#writing individual fragment input .py files to be run for RHF
x = 0
class_list = [frag1, frag2, frag3]   #fragmentation instances
fraglist = ['frag1', 'frag2', 'frag3']
mim_coeffs = [1, -1, 1]
for level in range(0, len(class_list)):
    os.mkdir(fraglist[level])
    os.chdir(fraglist[level])
    np.save(os.path.join('mim_co.npy'), mim_coeffs[level])
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
        "mol.atom = ", str(inputxyz), "\n", "mol.basis = '", basis_set, "'\n", "mol.build()\n"]
        f.writelines(build) 
        pyscf = open('../../../nicolefragment/Pyscf.py', 'r')
        lines = pyscf.readlines()
        
        #writes scf info depending on theory given
        emp = []
        grad_line = str()
        if i.qc_class.theory == 'RHF': 
            indices = [38, 39, 40, 41]
            grad_line = str('hf_scanner = scf.RHF(mol2).apply(grad.RHF).as_scanner()\n'+ '            e, g = hf_scanner(mol2)\n')
            for index in indices:
                emp.append(lines[index].lstrip())
        if i.qc_class.theory == 'MP2':
            indices = [45, 46, 47]
            grad_line = str("mp2_scanner = mp.MP2(scf.RHF(mol2)).nuc_grad_method().as_scanner()\n"+ "            e, g = mp2_scanner(mol2)\n")
            for index in indices:
                emp.append(lines[index].lstrip())
        if i.qc_class.theory == 'CISD':
            indices = [51, 52, 53]
            grad_line = str("ci_scanner = ci.CISD(scf.RHF(mol2)).nuc_grad_method().as_scanner()\n"+ "            e, g = ci_scanner(mol2)\n")
            for index in indices:
                emp.append(lines[index].lstrip())
        #dipole = ["mfx = scf.RHF(mol).run()\n, dipole = mfx.dip_moment(mol)\n"]
        f.writelines(x for x in emp)
        #f.writelines(dipole)
        pyscf.close()
        num_hess = ["#If not analytical hess, not do numerical below\n",
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
"            ", grad_line,
"            grad1 = g.flatten()\n",
"            mol.atom[atom][1][xyz] = mol.atom[atom][1][xyz]-2*step_size\n",
"            mol2 = gto.Mole()\n",
"            mol2.atom = mol.atom\n",
"            mol2.basis = mol.basis\n",
"            mol2.build()\n",
"            ", grad_line,
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


#files = os.listdir()
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
    os.chdir('../')

