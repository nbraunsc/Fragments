from Molecule import *
from Fragment import *
from fragmentation import *

""" MIM1 is one level of theory"""
def do_MIM1(deg, theory, basis, Molecule):
    frag = Fragmentation(Molecule)
    MIM1_energy, grad = frag.do_fragmentation(deg, theory, basis)
    norm = np.linalg.norm(grad)
    print('E(MIM1) =', MIM1_energy, 'Hartree')
    print('Grad(MIM1):', '\n', grad)
    print('Norm(grad) =', norm)
    return MIM1_energy, grad

"""MIM2 is two levels of theory"""
def do_MIM2(frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule):
    frag1 = Fragmentation(Molecule)
    E_high_highdeg, grad1 = frag1.do_fragmentation(frag_deg, high_theory, high_basis)
    
    frag2 = Fragmentation(Molecule)
    E_low_highdeg, grad2 = frag2.do_fragmentation(frag_deg, low_theory, low_basis)
    
    frag3 = Fragmentation(Molecule)
    E_infinite, grad3 = frag3.do_fragmentation(infinite_deg, low_theory, low_basis)
    
    MIM2_energy = E_high_highdeg - E_low_highdeg + E_infinite
    MIM2_grad = grad1 - grad2 + grad3
    norm = np.linalg.norm(MIM2_grad)
    print('E(MIM2) =', MIM2_energy, 'Hartree')
    print('Grad(MIM2):', '\n', MIM2_grad)
    print('Norm(grad) =', norm)
    return MIM2_energy, MIM2_grad

"""MIM3 is three levels of theory"""
def do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, Molecule):
    frag1 = Fragmentation(Molecule)  #must be here so it doesn't save old fragmentation
    E_high_highdeg, grad1 = frag1.do_fragmentation(frag_highdeg, high_theory, high_basis)
    
    frag2 = Fragmentation(Molecule)
    E_med_highdeg, grad2 = frag2.do_fragmentation(frag_highdeg, med_theory, med_basis)
    
    frag3 = Fragmentation(Molecule)
    E_med_meddeg, grad3 = frag3.do_fragmentation(frag_meddeg, med_theory, med_basis)
    
    frag4 = Fragmentation(Molecule)
    E_low_meddeg, grad4 = frag4.do_fragmentation(frag_meddeg, low_theory, low_basis)
    
    frag5 = Fragmentation(Molecule)
    E_infinite, grad5 = frag5.do_fragmentation(infinite_deg, low_theory, low_basis)
    
    MIM3_energy = E_high_highdeg - E_med_highdeg + E_med_meddeg - E_low_meddeg + E_infinite
    MIM3_grad = grad1 - grad2 + grad3 - grad4 + grad5
    norm = np.linalg.norm(MIM3_grad)
    print('E(MIM3) =', MIM3_energy, 'Hartree')
    print('Grad(MIM3) =', '\n', MIM3_grad)
    print('Norm(grad) =', norm)
    return MIM3_energy


if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin') #argument is input file name without any extension
        
    """do_MIM1(deg, theory, basis, Molecule)"""
    #do_MIM1(1, 'MP2', 'sto-3g', aspirin)        #uncomment to run MIM1
    
    """do_MIM2(frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule)"""
    do_MIM2(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', aspirin) #uncomment to run MIM2
    
    """do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, Molecule)"""
    #do_MIM3(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', 1, 'RHF', 'sto-3g', aspirin)     #uncomment to run MIM3
