import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import ray
import os
from pyscf.geomopt.berny_solver import optimize
from berny import Berny, geomlib
import matplotlib.pyplot as plt
import scipy.stats

np.set_printoptions(suppress=True, precision=5)

def global_props(frag_obj, step_size=0.001):
    """ This is a global func that is called within each MIM calculation.
    This is also doing the ray parallezation for the energy, gradient, hessians.

    Parameters 
    ----------
    frag_obj : Fragmentation object

    Returns 
    -------
    etot : float
        Global energy
    gtot : ndarray
        Global gradient
    htot : ndarray
        Global hessian
    """
    ray.init()
    #ray.init(address='nrlogin1:10.31.1.1')
    frags_id = ray.put(frag_obj)    #future for Fragmentation instance, putting in object store
    
    @ray.remote
    def get_frag_stuff(f,_frags):
        f_current = _frags.frags[f]
        return f_current.qc_backend(step_size=0.001)
    
    result_ids = [get_frag_stuff.remote(fi, frags_id) for fi in range(len(frag_obj.frags)) ]
    out = ray.get(result_ids)
    etot = 0
    gtot = 0
    htot = 0
    apt = 0
    for o in out:
        etot += o[0]
        gtot += o[1]
        htot += o[2]
        apt += o[3]
    ray.shutdown()
    return etot, gtot, htot, apt

def do_MIM1(deg, frag_type, theory, basis, Molecule, opt=False, step_size=0.001):
    """
    Parameters
    ----------
    deg : float
        Degree of fragmentation
    frag_type : str
        Type of fragmentation wanted. Either "distance" or "graphical"
    theory : str
        Theory wanted
    Molecule : Molecule class instance
    
    Returns
    -------
    etot : float
    gtot : ndarray
    htot : ndarray
    freq : ndarray
        Frequencies for the full molecule
    modes : ndarray 
        Normal modes for the full molecule
    """
    frag = fragmentation.Fragmentation(Molecule)
    frag.do_fragmentation(frag_type=str(frag_type), value=deg)
    frag.initalize_Frag_objects(theory=str(theory), basis=str(basis), qc_backend=Pyscf.Pyscf, step_size=0.001)
    #frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)

    if opt == True:
        #start of geom_optimization
        def opt_fnc(newcoords):
            for atom in range(0, len(newcoords)): #makes newcoords = self.molecule.atomtable
                x = list(newcoords[atom])
                frag.molecule.atomtable[atom][1:] = x
            
            frag.initalize_Frag_objects(theory=str(theory), basis=str(basis), qc_backend=Pyscf.Pyscf, step=step)
            etot, gtot, htot, apt = global_props(frag, step_size=step)
            return etot, gtot 
        
        frag.write_xyz(str(Molecule))
        os.path.abspath(os.curdir)
        os.chdir('../inputs/')
        optimizer = Berny(geomlib.readfile(os.path.abspath(os.curdir) + '/' + str(Molecule) + '.xyz'), debug=True)
        x = 0
        etot_opt = 0
        grad_opt = 0
        for geom in optimizer:
            x = x+1
            print("\n############# opt cycle:", x, "##################\n")
            solver = opt_fnc(geom.coords)
            optimizer.send(solver)
            etot_opt = solver[0]
            grad_opt = solver[1]
        relaxed = geom
        print("\n", "##########################", '\n', "#       Converged!       #", '\n', "##########################") 
        print('\n', "Energy = ", etot_opt)
        print('\n', "Converged_Gradient:", "\n", grad_opt)
    
    etot, gtot, htot, apt = global_props(frag, step_size=0.001)
    freq, modes = frag.mw_hessian(htot)
    pq = np.dot(apt.T, modes)   #shape 3x3N
    print("pq = ", pq.shape, pq)    
    pq_pq = np.dot(pq.T, pq)    #shape 3Nx3N
    print(pq_pq.shape)
    intense = np.diagonal(pq_pq)
    print("intensities in other units", intense)
    intense_kmmol = intense*42.2561
    #intense_kmmol = intense*42.2561*0.529177
    print("intensities in km/mol", intense_kmmol)
    print("Frequencies: ", freq, "cm-1")
    #print("Normal Modes: ", modes)
    print("Final converged energy = ", etot, "Hartree")
    print("Final gradient = ", '\n', gtot)
    #print("Final hessian = ", '\n', htot)
    print("Hessian shape = ", htot.shape)
    for i in range(0, len(freq)):
        print("Freq:", freq[i], "int :", intense_kmmol[i])
    
    #def model(position, width, height):
    #    return  height * np.sqrt(2*np.pi) * width * scipy.stats.norm.pdf(x, position, width) 
    def model(position, width, height):
        return  (height / scipy.stats.norm.pdf(position,position,width)) * scipy.stats.norm.pdf(x, position, width)

    for freq_i in range(6, len(freq)):
        position = freq[freq_i]
        x_min = position-300
        x_max = position+300
        height = intense_kmmol[freq_i]
        width = 150/(height/2)
        x=np.linspace(x_min, x_max, 100)
        gauss = model(position, width, height)
        plt.plot(x,gauss,color='red')
        plt.fill_between(x, gauss, color='red')
        #y=scipy.stats.norm.pdf(x, position, width)
        #y_scaled = y*(1/height)
        #if height > 3:
            #plt.plot(x,gauss,color='red')
            #plt.fill_between(x, gauss, color='red')
        #else:
        #    continue
    
    #y=scipy.stats.norm.pdf(x, mean, std)
    #plt.plot(x, y, color='coral')
    
    plt.xlabel('frequency (cm-1)')
    plt.ylabel('intensity (km/mol)')
    
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlim(4000, 500)
    plt.show()

    #plt.plot(freq, intense_kmmol)
    #plt.xlabel('frequency (cm-1)')
    #plt.ylabel('intensity (km/mol)')
    #plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()
    #plt.show()
    return etot, gtot, htot, freq, modes

def do_MIM2(frag_type, frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule,  opt=False):
    """
    MIM2 is two levels of theory with two levels of fragmentation.
    Parameters
    ----------
    frag_type : str
        Type of fragmentation wanted. Either "distance" or "graphical"
    frag_deg : float
        Degree of fragmentation
    high_theory : str
        higher level of theory
    high_basis : str
        larger basis set
    infinite_deg : float
        larger degree of fragmentation resultig in larger fragments
    low_theory : str
        lower level of theory
    low_basis : str
        smaller basis set
    Molecule : Molecule class instance
    opt : boolean
        False (default) for no geometry opt, True for geom_opt
    
    Returns
    -------
    MIM2_energy : float
        Global MIM2 energy
    MIM2_grad : ndarray
        Global MIM2 gradient
    MIM2_hess : ndarray
        Global MIM2 hessian
    """

    """ MIM high theory, small fragments"""
    frag1 = fragmentation.Fragmentation(Molecule)
    frag1.do_fragmentation(frag_type=str(frag_type), value=frag_deg)
    frag1.initalize_Frag_objects(theory=str(high_theory), basis=str(high_basis), qc_backend=Pyscf.Pyscf)
    #frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)
    
    """ MIM low theory, small fragments"""
    frag2 = fragmentation.Fragmentation(Molecule)
    frag2.do_fragmentation(frag_type=str(frag_type), value=frag_deg)
    frag2.initalize_Frag_objects(theory=str(low_theory), basis=str(high_basis), qc_backend=Pyscf.Pyscf)
    #frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)
    
    """ MIM low theory, large fragments (inifinte system)"""
    frag3 = fragmentation.Fragmentation(Molecule)
    frag3.do_fragmentation(frag_type=str(frag_type), value=infinite_deg)
    frag3.initalize_Frag_objects(theory=str(low_theory), basis=str(high_basis), qc_backend=Pyscf.Pyscf)
    #frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)
    
    if opt == True:
        #start of geom_optimization
        def opt_fnc(newcoords):
            for atom in range(0, len(newcoords)): #makes newcoords = self.molecule.atomtable
                x = list(newcoords[atom])
                frag1.molecule.atomtable[atom][1:] = x
            frag2.molecule.atomtable = frag1.molecule.atomtable 
            frag3.molecule.atomtable = frag1.molecule.atomtable 
            
            frag1.initalize_Frag_objects(theory=str(high_theory), basis=str(high_basis), qc_backend=Pyscf.Pyscf)
            frag2.initalize_Frag_objects(theory=str(low_theory), basis=str(high_basis), qc_backend=Pyscf.Pyscf)
            frag3.initalize_Frag_objects(theory=str(low_theory), basis=str(high_basis), qc_backend=Pyscf.Pyscf)
            
            MIM2_energy = 0
            MIM2_grad = np.zeros((frag1.molecule.natoms, 3))
            
            etot1, gtot1, htot1, apt1 = global_props(frag1, step=0.001)
            etot2, gtot2, htot2, apt2 = global_props(frag2, step=0.001)
            etot3, gtot3, htot3, apt3 = global_props(frag3, step=0.001)
            MIM2_energy = etot1 - etot2 + etot3
            MIM2_grad = gtot1 - gtot2 + gtot3
            return MIM2_energy, MIM2_grad
        
        frag1.write_xyz(str(Molecule))
        os.path.abspath(os.curdir)
        os.chdir('../inputs/')
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
    
    
    "Running energy, grad, hess, and apt build at optimized geometry"
    etot1, gtot1, htot1, apt1 = global_props(frag1, step=0.001)
    etot2, gtot2, htot2, apt2 = global_props(frag2, step=0.001)
    etot3, gtot3, htot3, apt3 = global_props(frag3, step=0.001)
    etot = etot1 - etot2 + etot3
    gtot = gtot1 - gtot2 + gtot3
    htot = htot1 - htot2 + htot3
    apt = apt1 - apt2 + apt3

    freq, modes = frag1.mw_hessian(htot)
    pq = np.dot(apt.T, modes)   #shape 3x3N
    print("pq = ", pq.shape, pq)    
    pq_pq = np.dot(pq.T, pq)    #shape 3Nx3N
    print(pq_pq.shape)
    intense = np.diagonal(pq_pq)
    print("intensities in other units", intense)
    intense_kmmol = intense*42.2561
    #intense_kmmol = intense*42.2561*0.529177
    print("intensities in km/mol", intense_kmmol)
    print("Frequencies: ", freq*.908, "cm-1")
    print("Final converged energy = ", etot, "Hartree")
    print("Final gradient = ", '\n', gtot)
    print("Hessian shape = ", htot.shape)
    for i in range(0, len(freq)):
        print("Freq:", freq[i], "int :", intense_kmmol[i])
    return etot, gtot, htot, apt, freq, intense_kmmol
        
    
def do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, Molecule):
    """
    NEED TO STILL IMPLEMENT MIM3 STUFF WITH THE IR STUFF
    MIM3 is three different levels of theory with three fragmentations.
    :frag_highdeg - smaller fragments
    :frag_meddeg - medium sized fragments
    :infinite_deg - large fragments (could be whole molecule)
    :theory and basis sets change level as well
    :Molecule - Molecule class object
    :str(Molecule) - the str(Molecule) of the Molecule class object without being a class, used in geomopt
    """
    frag1 = Fragmentation(Molecule)  
    frag1.do_fragmentation(frag_highdeg, high_theory, high_basis)
    E_high_highdeg, grad1 = frag1.do_geomopt(str(Molecule), high_theory, high_basis)
    frag1_hess, frag1_freq, frag1_vectors = frag1.compute_Hessian(high_theory, high_basis)
    
    frag2 = Fragmentation(Molecule)
    frag2.do_fragmentation(frag_highdeg, med_theory, med_basis)
    E_med_highdeg, grad2 = frag2.do_geomopt(str(Molecule), med_theory, med_basis)
    frag2_hess, frag2_freq, frag2_vectors = frag2.compute_Hessian(med_theory, med_basis)
    
    frag3 = Fragmentation(Molecule)
    frag3.do_fragmentation(frag_meddeg, med_theory, med_basis)
    E_med_meddeg, grad3 = frag3.do_geomopt(str(Molecule), med_theory, med_basis)
    frag3_hess, frag3_freq, frag3_vectors = frag3.compute_Hessian(med_theory, med_basis)
    
    frag4 = Fragmentation(Molecule)
    frag4.do_fragmentation(frag_meddeg, low_theory, low_basis)
    E_low_meddeg, grad4 = frag4.do_geomopt(str(Molecule), low_theory, low_basis)
    frag4_hess, frag4_freq, frag4_vectors = frag4.compute_Hessian(low_theory, low_basis)
    
    frag5 = Fragmentation(Molecule)
    frag5.do_fragmentation(infinite_deg, low_theory, low_basis)
    E_infinite, grad5 = frag5.do_geomopt(str(Molecule), low_theory, low_basis)
    frag5_hess, frag5_freq, frag5_vectors = frag5.compute_Hessian(low_theory, low_basis)
    
    MIM3_energy = E_high_highdeg - E_med_highdeg + E_med_meddeg - E_low_meddeg + E_infinite
    MIM3_grad = grad1 - grad2 + grad3 - grad4 + grad5
    MIM3_hess = frag1_hess - frag2_hess + frag3_hess - frag4_hess + frag5_hess
    norm = np.linalg.norm(MIM3_grad)
    print('E(MIM3) =', MIM3_energy, 'Hartree')
    print('Grad(MIM3) =', '\n', MIM3_grad)
    print('Hess(MIM3):', '\n', MIM3_hess)
    print('Norm(grad) =', norm)
    return MIM3_energy


if __name__ == "__main__":
    largermol = Molecule.Molecule()
    largermol.initalize_molecule('largermol')
        
    """do_MIM1(deg, frag_type,  theory, basis, Molecule, opt=False, step=0.001)"""
    do_MIM1(1.8, 'distance', 'RHF', 'sto-3g', largermol, opt=False, step_size=0.001)        #uncomment to run MIM1
    
    """do_MIM2(frag_type, frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule, opt=False)"""
    #do_MIM2('distance', 1.3, 'MP2', 'ccpvdz', 1.8, 'RHF', 'ccpvdz', largermol, opt=False) #uncomment to run MIM2
    
    """do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, Molecule)"""
    #do_MIM3(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', 1, 'RHF', 'sto-3g', largermol, 'ethanol')     #uncomment to run MIM3
