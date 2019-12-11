from fragclasses import *

def test_pie():
    """
    This is testing if the derivative recursive function works
    """
    aspirin = Molecule()
    aspirin.initalize_molecule()
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1)
    assert(runpie(frag.frag) == [{1, 3, 4, 5, 6}, {6}, {6}, {1, 4}, {1}, {1, 6}, {2, 6, 7}, {6}, {0, 1, 4}, {6}, {1, 2, 6}])

if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule()
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1)
    test_pie()
    