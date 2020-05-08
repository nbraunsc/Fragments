from .Molecule import *
from .Fragment import *
from .fragmentation import *

def do_MIM1(deg, theory, basis, Molecule, name):
    """
    MIM1 is only one level of fragmentation and one level of theory.
    :deg - degree of fragmentation
    :bsis - basis set
    :Molecule - Molecule class object
    :name - the name of the Molecule class object without being a class, used in geomopt
    """
    frag = Fragmentation(Molecule)
    frag.do_fragmentation(deg, theory, basis)
    MIM1_energy, grad = frag.do_geomopt(name, theory, basis)
    #MIM1_hess, MIM1_freq, MIM1_vectors = frag.compute_Hessian(theory, basis)
    #norm = np.linalg.norm(grad)
    print('E(MIM1) =', MIM1_energy, 'Hartree')
    print('Grad(MIM1):', '\n', grad)
    #print('Hess(MIM1):', '\n', MIM1_hess)
    #print('Frequencies:', '\n', MIM1_freq)
    #print('Norm(grad) =', norm)
    return MIM1_energy, grad    #, MIM1_hess, MIM1_freq, MIM1_vectors

def do_MIM2(frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule, name):
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
    frag1 = Fragmentation(Molecule)
    frag1.do_fragmentation(frag_deg, high_theory, high_basis)
    E_high_highdeg, grad1 = frag1.do_geomopt(name, high_theory, high_basis)
    frag1_hess, frag1_freq, frag1_vectors = frag1.compute_Hessian(high_theory, high_basis)
    
    frag2 = Fragmentation(Molecule)
    frag2.do_fragmentation(frag_deg, low_theory, low_basis)
    E_low_highdeg, grad2 = frag2.do_geomopt(name, low_theory, low_basis)
    frag2_hess, frag2_freq, frag2_vectors = frag2.compute_Hessian(low_theory, low_basis)
    
    frag3 = Fragmentation(Molecule)
    frag3.do_fragmentation(infinite_deg, low_theory, low_basis)
    E_infinite, grad3 = frag3.do_geomopt(name, low_theory, low_basis)
    frag3_hess, frag3_freq, frag3_vectors = frag3.compute_Hessian(low_theory, low_basis)
    
    MIM2_energy = E_high_highdeg - E_low_highdeg + E_infinite
    MIM2_grad = grad1 - grad2 + grad3
    MIM2_hess = frag1_hess - frag2_hess + frag3_hess
    
    ####################################################################
    # do i diagnolize hessian here? or for each frag1, frag2, frag3??? #
    ####################################################################

    norm = np.linalg.norm(MIM2_grad)
    print('E(MIM2) =', MIM2_energy, 'Hartree')
    print('Grad(MIM2):', '\n', MIM2_grad)
    print('Hess(MIM2):', '\n', MIM2_hess)
    print('Norm(grad) =', norm)
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
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin') #argument is input file name without any extension
        
    """do_MIM1(deg, theory, basis, Molecule)"""
    do_MIM1(1, 'MP2', 'sto-3g', aspirin, 'aspirin')        #uncomment to run MIM1
    
    """do_MIM2(frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule)"""
    #do_MIM2(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', aspirin, 'aspirin') #uncomment to run MIM2
    
    """do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, Molecule)"""
    #do_MIM3(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', 1, 'RHF', 'sto-3g', aspirin, 'aspirin')     #uncomment to run MIM3
