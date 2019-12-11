from fragclasses import *
#from tools import runpie
import unittest

class TestPie(unittest.TestCase):
    def test_pie(self):
        """
        This is testing if the derivative recursive function works
        """
        print(frag.frag)
        self.assertEqual(runpie(frag.frag), [{1, 3, 4, 5, 6}, {6}, {6}, {1, 4}, {1}, {1, 6}, {2, 6, 7}, {6}, {0, 1, 4}, {6}, {1, 2, 6}])

if __name__ == "__main__":
    aspirin = Molecule()
    frag = Fragmentation(aspirin)
    frag.do_fragmentation(1)
    unittest.main()

"""
this is not finding the frag list I need, do not know how to fix this is
"""
