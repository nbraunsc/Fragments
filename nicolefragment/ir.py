from ase import Atoms
from ase.io import read
from ase.visualize import view
from ase.vibrations import Infrared  
from ase.calculators.vasp import Vasp

def get_IR_ase():
    water = Atoms('H2O', positions=[[0, 0, .151], [0.758602, 0.000000, 0.504284], [0.758602, 0.000000,-0.504284]])
    #water = read('../inputs/water.xyz')
    calc = Vasp(prec='Accurate',
             ediff=1E-8,
             isym=0,
             idipol=4,       # calculate the total dipole moment
             dipol=water.get_center_of_mass(scaled=True),
             ldipol=True)
    water.set_calculator(calc)
    ir = Infrared(water)
    ir.run()
    ir.summary()


import numpy as np
from pyscf import gto, scf, ci, cc, mp, hessian, lib, grad, mcscf, dft
from pyscf.prop.freq import rhf, rks
from pyscf.prop.polarizability import rhf
from pyscf.grad.rhf import GradientsBasics
from pyscf.geomopt.berny_solver import optimize
from numpy import linalg as LA 
from scipy.linalg import orth

mol = gto.Mole()

#mol.atom = """
#C        0.0000010000     -0.7133160000      0.0000000000                 
#C        0.0000010000      0.7133160000      0.0000000000                 
#C        0.0000040000      3.5238760000      0.0000000000                 
#C        0.0000010000     -3.5238760000      0.0000000000                 
#C        1.2362690000     -1.4291120000      0.0000000000                 
#C       -1.2362670000      1.4291100000      0.0000000000                 
#C        1.2362690000      1.4291120000      0.0000000000                 
#C       -1.2362670000     -1.4291100000      0.0000000000                 
#C        2.4638520000      0.6807780000      0.0000000000                 
#C        2.4638520000     -0.6807780000      0.0000000000                 
#C       -2.4638520000     -0.6807780000      0.0000000000                 
#C       -2.4638520000      0.6807780000      0.0000000000                 
#C        1.2106350000      2.8329730000      0.0000000000                 
#C        1.2106400000     -2.8329740000      0.0000000000                 
#C       -1.2106370000     -2.8329720000      0.0000000000                 
#C       -1.2106420000      2.8329720000      0.0000000000                 
#H        2.1505210000      3.3797840000      0.0000000000                 
#H       -2.1505380000      3.3797640000      0.0000000000                 
#H        2.1505340000     -3.3797690000      0.0000000000                 
#H       -2.1505250000     -3.3797780000      0.0000000000                 
#H        3.4022100000     -1.2303750000      0.0000000000                 
#H        3.4022100000      1.2303760000      0.0000000000                 
#H       -3.4022040000     -1.2303860000      0.0000000000                 
#H       -3.4022040000      1.2303860000      0.0000000000                 
#H       -0.0000060000     -4.6105690000      0.0000000000                 
#H        0.0000100000      4.6105690000      0.0000000000
#"""
#mol.atom = [['O', (0.0, 0.0, 0.1519867318)], ['H', (-0.7631844707, 0.000000, -0.4446133658)], ['H', (0.7631844707, 0.000000, -0.4446133658)]]
mol.atom = [['O',(0, 0, 0)], ['H', (-0, 0.76181100, -0.59871000)], ['H', (-0, -0.76181100, -0.59871000)]]
mol.basis = 'ccpvdz'
mol.build()

#Hartree Fock Trial
mf = scf.RHF(mol).run()
energy = mf.kernel()
#mol_eq = optimize(mf)
hess = mf.Hessian().kernel()
#hess = mol_eq.Hessian().kernel()

#pyscf frequencies from dft
mfp = dft.RKS(mol).run()
w, modes_pyscf = rks.Freq(mfp).kernel()

dipole = mf.dip_moment(mol)
print(dipole)

mo_id = 2
dm_init_guess = [None]
def apply_field(E):
    """ This will apply an electric field in a specific direction for pyscf.

    Parameters
    ----------
    E : np array
        This is a 1D array of an x, y, z.  Put magintude of wanted E field in the position of the
        direction wanted.

    Returns
    -------
    mos : ndarray?
        This are the new mos in the core hamiltonian used for another SCF calculation.

    """

    mol.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
    h =(mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
      + np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3)))
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h
    mf.scf(dm_init_guess[0])
    dm_init_guess[0] = mf.make_rdm1()
    mo = mf.mo_coeff[:,mo_id]
    #if mo[23] < -1e-5:  # To ensure that all MOs have same phase
    #    mo *= -1
    return mo

from mendeleev import element

def num_diff(step):
    """ This is the function to use the apply_field() and run the finite difference for
    the whole system. Have yet to optimize/finish this funciton yet.
    """

    #mos = apply_field((step, 0, 0))
    #print("Mos = ", mos, mos.shape)
    #m = scf. RHF(mol).run()
    #e2 = scf. RHF(m).kernel()
    #g2 = grad.RHF(m).kernel()
   
    e_field = []
    dipole_moment = [0, 0, 0]
    for coord in range(0, 2):
        e_field = [0, 0, 0]
        e_field[coord] = step
        mos = apply_field(e_field)
        m = scf.RHF(mol).run()
        energy = m.kernel()
        #gradient = grad.RHF(m).kernel() 
        comp = (e-energy)/(2*step)
        dipole_moment[coord] = comp

    #print("Num dipole moment = ", dipole_moment)
    #print("Originial dipole moment = ", dipole)
    ###########################################
    #print("First grad = ", g)
    #print("Second grad = ", g2)
    #diff = np.subtract(g[0], g2[0])
    #print("Diff = ", diff)
    #ax = diff/0.001
    ###print("x tensor = ", ax, ax.shape)

def normal_modes(labels, xyz):
    """ Funciton to take in the hessian from pyscf calc and give frequenices 
    and normal modes.
    """
    #mass_matrix = np.zeros(hess.shape)  
    for row in range(0, len(labels)):
        for column in range(0, len(labels)):    
            x = element(labels[row])
            y = element(labels[column])
            z = x.atomic_weight*y.atomic_weight
            value = np.sqrt(z)**-1
            term = hess[row][column]*value
            hess[row][column] = term
            #np.fill_diagonal(mass_matrix[row][column], value)
            
    #adding 1/np.sqrt(amu) units to xyz coords
    mass_xyz = np.zeros(xyz.shape)
    mass_array = np.zeros((3))
    for atom in range(0, len(labels)):
        x = element(labels[atom])
        value = np.sqrt(x.atomic_weight)
        mass_array[atom] = value
        mass_xyz[atom] = xyz[atom]*(1/value)   #mass weighted coordinates
   
    reshape_mass_hess = hess.transpose(0, 2, 1, 3)
    x = reshape_mass_hess.reshape(reshape_mass_hess.shape[0]*reshape_mass_hess.shape[1],reshape_mass_hess.shape[2]*reshape_mass_hess.shape[3])
    e_values, e_vectors = LA.eigh(x)

    #start of unit conversion of freq
    factor = 1.8897259886**2*(4.3597482*10**-18)/(1.6603145*10**-27)/(1.0*10**-20)
    freq = (np.sqrt(e_values*factor))/(2*np.pi*2.9979*10**10)
    return e_vectors, freq, e_values, mass_xyz

def get_dipole(coords_new):
    mol2 = gto.Mole()
    mol2.atom = [['O', coords_new[0]], ['H', coords_new[1]], ['H', coords_new[2]]]
    mol2.basis = 'ccpvdz'
    mol2.build()
    mfx = scf.RHF(mol2).run()
    dipole1 = mfx.dip_moment(mol2)
    return dipole1

np.set_printoptions(suppress=True, precision=5)
def test_apt(step):
    """ xyz atomic displacement function.
    """
    xyz_org = np.zeros((3,3))
    labels = ['O', 'H', 'H']
    xyz_org[0] = [0.00000000, 0.00000000, 0.00000000]
    xyz_org[1] = [-0.00000000, 0.76181100, -0.59871000]
    xyz_org[2] = [-0.00000000, -0.76181100, -0.59871000]
    #xyz[0] = [0.0, 0.0, 0.1519867318]
    #xyz[1] = [-0.7631844707, 0.000000, -0.4446133658]
    #xyz[2] = [0.7631844707, 0.000000, -0.4446133658]
    
    modes, freq_cm, values, xyz = normal_modes(labels, xyz_org)
    apt = []
    for atom in range(0, len(labels)):  #atom interation
        storing_vec = np.zeros((3,3))
        for comp in range(0, len(xyz[atom])):   #xyz interation
            dip1 = get_dipole(xyz)
            value = xyz[atom][comp]+step
            xyz[atom][comp] = value
            dip2 = get_dipole(xyz)
            xyz[atom][comp] = xyz[atom][comp]-step
            vec = (dip1 - dip2)/step
            storing_vec[comp] = vec
        x = storing_vec.T
        apt.append(x)
    intensity = []
    
    px = np.vstack(apt)
    pq = np.dot(px.T, modes)
    print("pq = ", pq.shape, pq)
    pq_pq = np.dot(pq.T, pq)
    print(pq_pq.shape)
    intense = np.diagonal(pq_pq)
    print("intensities in other units", intense)
    intense_kmmol = intense*42.2561*0.529177
    print("intensities in km/mol", intense_kmmol)
    webmo_int = [77.298, 19.019, 51.071]
    print('Webmo intensites', webmo_int)
    diff_int = np.array(intense_kmmol[6:]) - webmo_int
    factor_int = webmo_int/np.array(intense_kmmol[6:])
    print("apt int factors", factor_int)
    print("intensity diff", diff_int)
    print("frequenies:", freq_cm)
    

def testing(step):
    """ this one displaces the coords along nomral mode, 
    need to match these for the apts so I can do the LA projection.
    This one is working for frequencies and intensities are of decent comparison.
    """
    xyz_org = np.zeros((3,3))
    labels = ['O', 'H', 'H']
    #xyz_org[0] = [0.0, 0.0, 0.1519867318]
    #xyz_org[1] = [-0.7631844707, 0.000000, -0.4446133658]
    #xyz_org[2] = [0.7631844707, 0.000000, -0.4446133658]
    #xyz_org[0] = [0, 0, 0]
    #xyz_org[1] = [-0.74867500, -0.00000000, -0.57850700]
    #xyz_org[2] = [0.74867500, 0.00000000, -0.57850700]
    xyz_org[0] = [0.00000000, 0.0000000, 0.00000000]
    xyz_org[1] = [-0.00000000, 0.76181100, -0.59871000]
    xyz_org[2] = [-0.00000000, -0.76181100, -0.59871000]
    
    modes, freq_cm, values, xyz = normal_modes(labels, xyz_org) #xyz are mass-weighted xyz coords
    intensity = []
    for i in range(0, len(modes)):  #normal mode interations
        print("\n################# Normal Mode", i, "#################")
        modes[i].reshape(1, 3*len(labels))
        
        coords = xyz.reshape(1,3*len(labels))
        #displace coord in postive direction
        coords1 = coords + step*(modes[i])

        #first dipole moment calculation
        coords_new = coords1.reshape(len(labels),3)
        print("coords in xyz format", coords_new)
        mol2 = gto.Mole()
        mol2.atom = [[labels[0], coords_new[0]], [labels[1], coords_new[1]], [labels[2], coords_new[2]]]
        mol2.basis = 'ccpvdz'
        mol2.build()
        mfx = scf.RHF(mol2).run()
        dipole1 = mfx.dip_moment(mol2)
        #change coords back and subtract displacement
        coords2 = coords - step*(modes[i])
        

        #second dipole moment calc
        coords_new2 = coords2.reshape(len(labels),3)
        mol3 = gto.Mole()
        mol3.atom = [[labels[0], coords_new2[0]], [labels[1], coords_new2[1]], [labels[2], coords_new2[2]]]
        mol3.basis = 'ccpvdz'
        mol3.build()
        mfy = scf.RHF(mol3).run()
        dipole2 = mfy.dip_moment(mol3)
        storing = 0
        for j in range(0, 3):   #x, y, z iteration
            diff = (dipole1[j] - dipole2[j])/(2*step)
            #diff = 0.52917721067121*(dipole1[j] - dipole2[j])/(2*step)
            storing += diff**2
        intensity.append(storing)
    #unit conversion of IR intensities
    #D^2/A^2/amu -> km/mol
    intensity_conv = list(np.array(intensity)*42.2561)

    for i in range(0, len(freq_cm)):
        print("Frequency = ", freq_cm[i], "(cm^-1)", "with intensity = ", intensity_conv[i], "(km/mol)")

    print("Pyscf frequenices", w)
    #print("modes", modes_pyscf)
    
    webmo_dipole = 2.0441   #Debye
    webmo_freq = [1833.92, 3806.93, 3896.73]
    webmo_int = [77.298, 19.019, 51.071]
    code_psi = freq_cm[6:] - webmo_freq
    print("Frequency difference:", code_psi)
    #py_psi = w[6:] - psi_freq
    #print("Pyscf with webmo diff", py_psi)
    diff = energy+76.0259957569
    print("energy differences between mine and webmo:", diff, "Hartrees")
    print("Intensity difference:", np.array(intensity_conv[6:]) - np.array(webmo_int), "km/mol")
    int_factors = webmo_int/np.array(intensity_conv[6:])
    print("intensity factors:", int_factors)
    freq_factors = webmo_freq/(freq_cm[6:])
    print("frequencies factors:", freq_factors)


if __name__ == "__main__":
    #get_IR()
    #normal_modes()
    #num_diff(0.01)
    #testing(0.0001)
    test_apt(0.0001)
