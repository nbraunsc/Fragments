from Molecule import *
from Fragment import *
from fragmentation import *

""" MIM1 is one level of theory"""
def do_MIM1(deg, theory, basis, Molecule):
    frag = Fragmentation(Molecule)
    MIM1_energy, grad = frag.do_fragmentation(deg, theory, basis)
    ##xyz = frag.print_fullxyz()
    print('E(MIM1) =', MIM1_energy, 'Hartree')
    #grad = []
    return MIM1_energy#, grad

"""MIM2 is two levels of theory"""
def do_MIM2(frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule):
    frag1 = Fragmentation(Molecule)
    E_high_highdeg = frag1.do_fragmentation(frag_deg, high_theory, high_basis)
    
    frag2 = Fragmentation(Molecule)
    E_low_highdeg = frag2.do_fragmentation(frag_deg, low_theory, low_basis)
    
    frag3 = Fragmentation(Molecule)
    E_infinite = frag3.do_fragmentation(infinite_deg, low_theory, low_basis)
    
    MIM2_energy = E_high_highdeg - E_low_highdeg + E_infinite
    print('E(MIM2) =', MIM2_energy, 'Hartree')
    return MIM2_energy

"""MIM3 is three levels of theory"""
def do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, Molecule):
    frag1 = Fragmentation(Molecule)  #must be here so it doesn't save old fragmentation
    E_high_highdeg = frag1.do_fragmentation(frag_highdeg, high_theory, high_basis)
    
    frag2 = Fragmentation(Molecule)
    E_med_highdeg = frag2.do_fragmentation(frag_highdeg, med_theory, med_basis)
    
    frag3 = Fragmentation(Molecule)
    E_med_meddeg = frag3.do_fragmentation(frag_meddeg, med_theory, med_basis)
    
    frag4 = Fragmentation(Molecule)
    E_low_meddeg = frag4.do_fragmentation(frag_meddeg, low_theory, low_basis)
    
    frag5 = Fragmentation(Molecule)
    E_infinite = frag5.do_fragmentation(infinite_deg, low_theory, low_basis)
    
    MIM3_energy = E_high_highdeg - E_med_highdeg + E_med_meddeg - E_low_meddeg + E_infinite
    print('E(MIM3) =', MIM3_energy, 'Hartree')
    return MIM3_energy


if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin') #argument is input file name without any extension
        
    """do_MIM1(deg, theory, basis, Molecule)"""
    do_MIM1(1, 'RHF', 'sto-3g', aspirin)        #uncomment to run MIM1
    
    """do_MIM2(frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, Molecule)"""
    #do_MIM2(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', aspirin) #uncomment to run MIM2
    
    """do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, Molecule)"""
    #do_MIM3(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', 1, 'RHF', 'sto-3g', aspirin)     #uncomment to run MIM3