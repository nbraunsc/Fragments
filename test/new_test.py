import nicolefragment
from nicolefragment import Molecule, fragmentation, Fragment, cov_rad, runpie, Pyscf
import numpy as np

def test_energy():
    """
    This is testing the overall energy by taking energies of fragments multiplied by their coefficents and then added up.  Should be the same energy as the full  molecule
    """
    carbonylavo = Molecule.Molecule()
    carbonylavo.initalize_molecule('carbonylavo')
    frag = fragmentation.Fragmentation(carbonylavo)
    frag.do_fragmentation(frag_type='distance', value=3)
    frag.initalize_Frag_objects(theory='RHF', basis='sto-3g', qc_backend=Pyscf.Pyscf)
    
    import ray
    ray.init()

    frags_id = ray.put(frag)    #future for Fragmentation instance, putting in object store

    @ray.remote
    def get_frag_stuff(f,_frags):
        f_current = _frags.frags[f]
        return f_current.qc_backend()
    result_ids = [get_frag_stuff.remote(fi, frags_id) for fi in range(len(frag.frags)) ]
    out = ray.get(result_ids)
    etot = 0
    gtot = 0
    for o in out:
        etot += o[0]
        gtot += o[1]
    value = -228.107654514819 - etot
    assert(value <= 1.0e-10) #will sometimes fail bec last decimal point is wrong

    #grad = np.array([[-4.88387004e-02, -1.15014666e-02,  4.15615023e-03], [5.62377195e-04, -2.52339637e-02,  4.79196023e-04], [2.60218256e-02,  8.30145767e-03, -1.89628820e-03], [1.97415592e-02,  3.21346632e-03, -3.50455438e-03], [-8.89716530e-05,  1.93243643e-03,  8.30002083e-03], [8.49181720e-05, -1.34662690e-03, -5.68122938e-03], [-1.61935486e-03, -5.54725087e-03, -5.90848200e-03], [-4.36362025e-04, -3.36642959e-03,  7.07457189e-03], [5.73354468e-03,  2.88104893e-02, -1.20250127e-02], [-2.92184536e-03,  7.72142070e-04, 8.00791311e-03], [4.08617352e-03, -3.05903414e-03,  7.75071221e-03], [2.52788351e-03,  1.12555968e-03, -5.74655869e-03], [-4.85304761e-03,  5.89922041e-03, -1.00643895e-03]])
     #assert np.array_equal(grad <= np.full((carbonylavo.natoms, 3), 1.0e-10))
    #grad_diff = np.allclose([gtot, grad], [1, 1e-10])
    #assert grad_diff == True

if __name__ == "__main__":
    carbonylavo = Molecule.Molecule()
    carbonylavo.initalize_molecule('carbonylavo')
    frag = fragmentation.Fragmentation(carbonylavo)
    frag.do_fragmentation(frag_type='distance', value=3)
    frag.initalize_Frag_objects(theory='RHF', basis='sto-3g', qc_backend=Pyscf)
    
    import ray
    ray.init()

    frags_id = ray.put(frag)    #future for Fragmentation instance, putting in object store

    @ray.remote
    def get_frag_stuff(f,_frags):
        f_current = _frags.frags[f]
        return f_current.qc_backend()
    result_ids = [get_frag_stuff.remote(fi, frags_id) for fi in range(len(frag.frags)) ]
    out = ray.get(result_ids)
    etot = 0
    gtot = 0
    for o in out:
        etot += o[0]
        gtot += o[1]
