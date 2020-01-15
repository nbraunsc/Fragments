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
    frag.do_fragmentation(1, 'RHF', 'sto-3g')
    print(frag.total)
    assert(frag.total == -636.6280880465249)
    

if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin')
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1, 'RHF', 'sto-3g')
    test_pie()
    test_energy()
    
