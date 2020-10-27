### Don't need this code ###

from pyscf import gto
import numpy

mol = gto.M(
    atom = [
        ['O' , 0. , 0.     , 0],
        ['H' , 0. , -0.757 , 0.587],
        ['H' , 0. ,  0.757 , 0.587]],
    basis = '631g')

mf = mol.RHF().run()
# The structure of h is
# h[Atom_1, Atom_2, Atom_1_XYZ, Atom_1_XYZ]
h = mf.Hessian().kernel()
print(h[0], 'first one')
print(2*h[0])

mf = mol.apply('UKS').x2c().run()
h = mf.Hessian().kernel()
