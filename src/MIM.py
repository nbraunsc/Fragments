from Molecule import *
from Fragment import *
from fragmentation import *

def do_MIM1(deg, theory, basis):
    MIM1_energy = frag.do_fragmentation(deg, theory, basis)
    print(MIM1_energy)
    return MIM1_energy

def do_MIM2(frag_deg, high_theory, high_basis, infinite_deg, low_theory, low_basis):
    E_high_highdeg = frag.do_fragmentation(frag_deg, high_theory, high_basis)
    E_low_highdeg = frag.do_fragmentation(frag_deg, low_theory, low_basis)
    E_infinite = frag.do_fragmentation(infinite_deg, low_theory, low_basis)
    MIM2_energy = E_high_highdeg - E_low_highdeg + E_infinite
    print(MIM2_energy)
    return MIM2_energy

def do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis):
    E_high_highdeg = frag.do_fragmentation(frag_highdeg, high_theory, high_basis)
    E_med_highdeg = frag.do_fragmentation(frag_highdeg, med_theory, med_basis)
    E_med_meddeg = frag.do_fragmentation(frag_meddeg, med_theory, med_basis)
    E_low_meddeg = frag.do_fragmentation(frag_meddeg, low_theory, low_basis)
    E_infinite = frag.do_fragmentation(infinite_deg, low_theory, low_basis)
    MIM3_energy = E_high_highdeg - E_med_highdeg + E_med_meddeg - E_low_meddeg + E_infinite
    print(MIM3_energy)
    return MIM3_energy


if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule()
    frag = Fragmentation(aspirin)
    #do_MIM1(1, 'RHF', 'sto-3g')
    do_MIM2(1, 'MP2', 'sto-3g', 1, 'RHF', 'sto-3g')
    #do_MIM3(frag_highdeg, high_theory, high_basis, frag_meddeg, med_theory, med_basis, infinite_deg, low_theory, low_basis)
