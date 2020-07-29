import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import ray

np.set_printoptions(suppress=True, precision=5)

def global_props(frag_obj):
    ray.init()
    frags_id = ray.put(frag_obj)    #future for Fragmentation instance, putting in object store
    
    @ray.remote
    def get_frag_stuff(f,_frags):
        f_current = _frags.frags[f]
        return f_current.qc_backend()
    
    result_ids = [get_frag_stuff.remote(fi, frags_id) for fi in range(len(frag_obj.frags)) ]
    out = ray.get(result_ids)
    etot = 0
    gtot = 0
    htot = 0
    for o in out:
        etot += o[0]
        gtot += o[1]
        htot += o[2]
    ray.shutdown()
    return etot, gtot, htot

def do_MIM1(deg, frag_type, theory, basis, Molecule):
    """
    MIM1 is only one level of fragmentation and one level of theory.
   
    Parameters
    ----------
    deg : float
        Degree of fragmentation
    frag_type : str
        Type of fragmentation wanted. Either "distance" or "graphical"
    theory : str
        Theory wanted
    Molecule : Molecule class instance
    name : str 
        String of the molecule class instance
    
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
    frag.initalize_Frag_objects(theory=str(theory), basis=str(basis), qc_backend=Pyscf.Pyscf)
    #frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)
    
    etot, gtot, htot = global_props(frag)
    freq, modes = frag.mw_hessian(htot)
    print("Frequencies: ", freq, "cm-1")
    print("Normal Modes: ", modes)
    print("Final converged energy = ", etot, "Hartree")
    print("Final gradient = ", '\n', gtot)
    #print("Final hessian = ", '\n', htot)
    print("Hessian shape = ", htot.shape)
    return etot, gtot, htot, freq, modes

def do_MIM2(frag_type, frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule):
    """
    MIM2 is two levels of theory with two levels of fragmentation.
    :frag_deg - smaller fragments
    :infinite_deg - larger fragments (could be whole molecule)
    :high_theory - higher level of theory
    :low_theory - lower level of theory
    :high and low basis follow same trend as theory
    :Molecule - Molecule class object
    :name - the name of the Molecule class object without being a class, used in geomopt
    """

    """ MIM high theory, small fragments"""
    frag1 = fragmentation.Fragmentation(Molecule)
    frag1.do_fragmentation(frag_type=str(frag_type), value=frag_deg)
    frag1.initalize_Frag_objects(theory=str(high_theory), basis=str(high_basis), qc_backend=Pyscf.Pyscf)
    #frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)
    etot1, gtot1, htot1 = global_props(frag1)
    freq1, modes1 = frag1.mw_hessian(htot1)
    
    """ MIM low theory, small fragments"""
    frag2 = fragmentation.Fragmentation(Molecule)
    frag2.do_fragmentation(frag_type=str(frag_type), value=frag_deg)
    frag2.initalize_Frag_objects(theory=str(low_theory), basis=str(low_basis), qc_backend=Pyscf.Pyscf)
    #frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)
    etot2, gtot2, htot2 = global_props(frag2)
    freq2, modes2 = frag2.mw_hessian(htot2)
    
    """ MIM low theory, large fragments (inifinte system)"""
    frag3 = fragmentation.Fragmentation(Molecule)
    frag3.do_fragmentation(frag_type=str(frag_type), value=infinite_deg)
    frag3.initalize_Frag_objects(theory=str(low_theory), basis=str(low_basis), qc_backend=Pyscf.Pyscf)
    #frag.qc_params(frag_index=[], qc_backend, theory, basis, spin=0, tol=0, active_space=0, nelec_alpha=0 nelec_beta=0, max_memory=0)
    etot3, gtot3, htot3 = global_props(frag3)
    freq3, modes3 = frag3.mw_hessian(htot3)
    
    MIM2_energy = etot1 - etot2 + etot3
    MIM2_grad = gtot1 - gtot2 + gtot3
    MIM2_hess = htot1 - htot2 + htot3
    print("Frequency 1:", freq1)
    print("Frequency 2:", freq2)
    print("Frequency 3:", freq3)
    return MIM2_energy, MIM2_grad, MIM2_hess
        
    
def do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, Molecule, name):
    """
    MIM3 is three different levels of theory with three fragmentations.
    :frag_highdeg - smaller fragments
    :frag_meddeg - medium sized fragments
    :infinite_deg - large fragments (could be whole molecule)
    :theory and basis sets change level as well
    :Molecule - Molecule class object
    :name - the name of the Molecule class object without being a class, used in geomopt
    """
    frag1 = Fragmentation(Molecule)  
    frag1.do_fragmentation(frag_highdeg, high_theory, high_basis)
    E_high_highdeg, grad1 = frag1.do_geomopt(name, high_theory, high_basis)
    frag1_hess, frag1_freq, frag1_vectors = frag1.compute_Hessian(high_theory, high_basis)
    
    frag2 = Fragmentation(Molecule)
    frag2.do_fragmentation(frag_highdeg, med_theory, med_basis)
    E_med_highdeg, grad2 = frag2.do_geomopt(name, med_theory, med_basis)
    frag2_hess, frag2_freq, frag2_vectors = frag2.compute_Hessian(med_theory, med_basis)
    
    frag3 = Fragmentation(Molecule)
    frag3.do_fragmentation(frag_meddeg, med_theory, med_basis)
    E_med_meddeg, grad3 = frag3.do_geomopt(name, med_theory, med_basis)
    frag3_hess, frag3_freq, frag3_vectors = frag3.compute_Hessian(med_theory, med_basis)
    
    frag4 = Fragmentation(Molecule)
    frag4.do_fragmentation(frag_meddeg, low_theory, low_basis)
    E_low_meddeg, grad4 = frag4.do_geomopt(name, low_theory, low_basis)
    frag4_hess, frag4_freq, frag4_vectors = frag4.compute_Hessian(low_theory, low_basis)
    
    frag5 = Fragmentation(Molecule)
    frag5.do_fragmentation(infinite_deg, low_theory, low_basis)
    E_infinite, grad5 = frag5.do_geomopt(name, low_theory, low_basis)
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
    carbonylavo = Molecule.Molecule()
    carbonylavo.initalize_molecule('carbonylavo')
        
    """do_MIM1(deg, frag_type,  theory, basis, Molecule)"""
    #do_MIM1(1.8, 'distance', 'RHF', 'sto-3g', carbonylavo)        #uncomment to run MIM1
    
    """do_MIM2(frag_type, frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule)"""
    do_MIM2('distance', 1.8, 'RHF', '631g', 3, 'RHF', 'sto-3g', carbonylavo) #uncomment to run MIM2
    
    """do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, Molecule)"""
    #do_MIM3(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', 1, 'RHF', 'sto-3g', aspirin, 'aspirin')     #uncomment to run MIM3
