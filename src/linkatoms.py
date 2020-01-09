import numpy as np

def add_linkatoms(atom1, attached_atom, molecule):
    atom1_element = molecule.atomtable[atom1][0]
    attached_atom_element = molecule.atomtable[attached_atom][0]
    cov_atom1 = molecule.covrad[atom1_element][0]
    cov_attached_atom = molecule.covrad[attached_atom_element][0]
    atom_xyz = np.array(molecule.atomtable[atom1][1:])
    attached_atom_xyz = np.array(molecule.atomtable[attached_atom][1:])
    vector = attached_atom_xyz - atom_xyz
    dist = np.linalg.norm(vector)
    h = 0.32
    factor = (h + cov_atom1)/(cov_atom1 + cov_attached_atom)
    new_xyz = list(factor*vector+atom_xyz)
    new_xyz.insert(0, 'H')
    return new_xyz


