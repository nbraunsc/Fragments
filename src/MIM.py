from Molecule import *
from Fragment import *
from fragmentation import *

def do_MIM1(deg, theory, basis, file_name):
    frag = Fragmentation(file_name)
    MIM1_energy = frag.do_fragmentation(deg, theory, basis)
    print('MIM1_Energy = ', MIM1_energy)
    return MIM1_energy

def do_MIM2(frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, file_name):
    frag1 = Fragmentation(file_name)
    frag2 = Fragmentation(file_name)
    frag3 = Fragmentation(file_name)
    E_high_highdeg = frag1.do_fragmentation(frag_deg, high_theory, high_basis)
    E_low_highdeg = frag2.do_fragmentation(frag_deg, low_theory, low_basis)
    E_infinite = frag3.do_fragmentation(infinite_deg, low_theory, low_basis)
    MIM2_energy = E_high_highdeg - E_low_highdeg + E_infinite
    print('MIM2_Energy = ', MIM2_energy)
    return MIM2_energy

def do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, file_name):
    frag1 = Fragmentation(file_name)  #must be here so it doesn't save old fragmentation
    frag2 = Fragmentation(file_name)
    frag3 = Fragmentation(file_name)
    frag4 = Fragmentation(file_name)
    frag5 = Fragmentation(file_name)
    E_high_highdeg = frag1.do_fragmentation(frag_highdeg, high_theory, high_basis)
    E_med_highdeg = frag2.do_fragmentation(frag_highdeg, med_theory, med_basis)
    E_med_meddeg = frag3.do_fragmentation(frag_meddeg, med_theory, med_basis)
    E_low_meddeg = frag4.do_fragmentation(frag_meddeg, low_theory, low_basis)
    E_infinite = frag5.do_fragmentation(infinite_deg, low_theory, low_basis)
    MIM3_energy = E_high_highdeg - E_med_highdeg + E_med_meddeg - E_low_meddeg + E_infinite
    print('MIM3_Energy = ', MIM3_energy)
    return MIM3_energy


if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin') #argument is input file name without any extension
        
    #do_MIM1(deg, theory, basis, file_name):
    do_MIM1(1, 'RHF', 'sto-3g', 'aspirin') #comment out this line to do MIM1
    
    #do_MIM2(frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis, file_name)
    #do_MIM2(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', 'aspirin') #comment out this line to do MIM2
    
    #do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis, file_name)
    #do_MIM3(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g', 1, 'RHF', 'sto-3g', 'aspirin')     #comment out this line to do MIM3
