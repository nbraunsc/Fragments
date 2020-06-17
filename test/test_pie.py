from Molecule import *
from fragmentation import *
from Fragment import *

def test_pie():
    """
    This is testing if the derivative recursive function works
    """
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin')
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1, 'RHF', 'sto-3g')
    assert(runpie(frag.unique_frag)[0] == [{1, 3, 4, 5, 6}, {6}, {6}, {1, 4}, {1}, {1, 6}, {2, 6, 7}, {2, 6}, {0, 1, 4}, {1}, {1, 2, 6}])

    """ coefficent test """
    assert(runpie(frag.unique_frag)[1] == [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1])

def test_energy():
    """
    This is testing the overall energy by taking energies of fragments multiplied by their coefficents and then added up.  Should be the same energy as the full  molecule
    """
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin')
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(frag_type='graphical', value=1)
    frag.initalize_Frag_objects(theory='RHF', basis='sto-3g', qc_backend=Pyscf)
    #e, g = frag.energy_gradient('RHF', 'sto-3g', frag.moleculexyz)
    value = -636.6280880465254 - e
    assert(value <= 1.0e-10) #will sometimes fail bec last decimal point is wrong

def test_geomopt():
    """
    Testing the optimizied total energy
    """
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin')
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1, 'RHF', 'sto-3g')
    frag.do_geomopt('aspirin', 'RHF', 'sto-3g')
    value = -636.6608658120065 - frag.etot_opt
    assert(value <= 1.0e-10)


if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin')
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(frag_type='graphical', value=1)
    frag.initalize_Frag_objects(theory='RHF', basis='sto-3g', qc_backend=Pyscf)
    test_pie()
    test_energy()
    #test_geomopt()

#Need to write tests to test the first gradient, the optimized gradient, and the different MIM levels (or maybe just MIM2)
