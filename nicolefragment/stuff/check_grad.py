### Don't need this python script for MIM code ###

from Molecule import *
from Fragment import *
from fragmentation import *
from MIM import *

class Numerical_Grad():
    """
    Heads up this is very slow!!! class that builds the numerical derivative and checks against the analytical derivative using finite difference.
    """

    def __init__(self, molecule):
        self.molecule = molecule
        self.a_energy = int()
        self.a_grad = []
        self.num_energy = int()
        self.num_grad = []

    def an_grad(self, deg, theory, basis):
        self.a_energy, self.a_grad = do_MIM1(deg, theory, basis, self.molecule)

    def num_gradient(self, deg, theory, basis, step):
        length = len(self.molecule.atomtable)
        self.num_grad = np.zeros((length,3))
        for i in range(0, length):
            for j in range(1, 4):
                self.molecule.atomtable[i][j] = self.molecule.atomtable[i][j]+step
                e = do_MIM1(deg, theory, basis, self.molecule)[0]
                factor = (e - self.a_energy)/(2*step)
                self.num_grad[i][j-1] = factor
                self.molecule.atomtable[i][j] = self.molecule.atomtable[i][j]-step
        
        diff = np.subtract(self.a_grad, self.num_grad)
        print(self.molecule.atomtable)
        print(self.a_grad, 'a grad')
        print(self.num_grad, 'num grad')
        print(diff, 'difference')

if __name__ == "__main__":
    aspirin = Molecule()
    aspirin.initalize_molecule('aspirin')
    num = Numerical_Grad(aspirin)
    num.an_grad(1, 'RHF', 'sto-3g')
    num.num_gradient(1, 'RHF', 'sto-3g', 0.00001)
