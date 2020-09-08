
import nicolefragment
from nicolefragment import runpie, Molecule, fragmentation, Fragment, Pyscf
import numpy as np
import os

#starting fragmentation code
adamantane = Molecule.Molecule()
adamantane.initalize_molecule('adamantane')
frag = fragmentation.Fragmentation(adamantane)
frag.do_fragmentation(frag_type='distance', value=1.3)
frag.initalize_Frag_objects(theory='RHF', basis='sto-3g', qc_backend=Pyscf.Pyscf, step_size=0.001)
