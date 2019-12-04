from fragclasses import *
import unittest

class TestPie(unittest.TestCase):
    def test_pie(self):
        """
        This is testing if the derivative recursive function works
        """
        data = frag.frag
        self.assertEqual(runpie(data), [{1, 3, 4, 5, 6}, {6}, {6}, {1, 4}, {1}, {1, 6}, {2, 6, 7}, {6}, {0, 1, 4}, {6}, {1, 2, 6}])

if __name__ == "__main__":
    aspirin = Molecule()
    frag = Fragmentation(aspirin)
    unittest.main()

"""
this is adding in another empty list so test is failing, do not know how to fix this is
"""
