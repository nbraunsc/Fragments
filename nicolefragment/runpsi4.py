import psi4
import numpy as np

def do_psi4(input_xyz, theory, basis, name):
    geom = psi4.geometry(input_xyz)
    #psi4.set_options({"SCF_TYPE": "hf","BASIS": "cc-pVDZ"})
    #scf_e = psi4.energy('SCF')
    e = psi4.energy(theory + "/" + basis, molecule=geom)
    #g = psi4.gradient(theory)
    return e #, g
