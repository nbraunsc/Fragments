import pickle
import os
import glob
import dill
import numpy as np
from mendeleev import element

os.chdir('to_run')
levels = os.listdir()

e = 0
g = 0
h = 0
apt = 0
apt_grad = 0
step = 0.01

for level in levels:
    os.chdir(level)
    frags = glob.glob('*.dill')
    for i in frags:
        #undill and run e, g, hess, apt etc
        infile = open(i, 'rb')
        new_class = dill.load(infile)
        print("Fragment ID:", level, i)
        e += new_class.energy
        g += new_class.grad
        h += new_class.hessian
        apt += new_class.apt 
        #apt_grad += new_class.aptgrad
        infile.close()
    os.chdir('../')

os.chdir('frag1')
infile = open('fragment0.dill', 'rb')
new_class = dill.load(infile)
freq, modes_unweight, M = new_class.mw_hessian(h)

################### All Normal mode testing for IR intensites ##########################

modes = np.dot(M, modes_unweight)   #mass-weighted normal modes
mole_coords = new_class.molecule.atomtable
labels = []
for i in range(0, len(mole_coords)):
    label = mole_coords[i][0]
    mole_coords[i].remove(label)
    labels.append(label)
atoms = np.array(labels)
coords = np.array(mole_coords).flatten()

x = np.indices(modes.shape)[1]
apt_norm = np.zeros((modes.shape[0], 3))

def normal_disp(modes_list):
    for i in modes_list:
        print("mode:", i)
        pos_list = []
        neg_list = []
        positive_coords = coords + step*modes[i] #getting new global coords for displacement along normal mode
        pos = positive_coords.reshape((new_class.molecule.natoms,3))
        neg_coords = coords - step*modes[i] #getting new global coords for displacement along normal mode
        neg = neg_coords.reshape((new_class.molecule.natoms,3))
        
        for row in range(0, new_class.molecule.natoms):
            pos_list.append(list(pos[row]))
            neg_list.append(list(neg[row]))
            pos_list[row].insert(0, labels[row])
            neg_list[row].insert(0, labels[row])
        new_class.molecule.atomtable = pos_list #setting positive disp global coords 
        inputxyz = new_class.build_xyz()    #getting pyscf xyz input format
        dip = new_class.qc_class.get_dipole(inputxyz)   #getting dipole with new coords
        new_class.molecule.atomtable = neg_list  #setting negative disp  global coords 
        inputxyz = new_class.build_xyz()    #getting pyscf xyz input format
        dip2 = new_class.qc_class.get_dipole(inputxyz)   #getting dipole with new coords
        apt_comp = (dip-dip2)/2*step    #3x1 vector
        apt_norm[i] = apt_comp

normal_disp(x[0])
print(apt_norm)
norm_freq = np.dot(apt_norm, apt_norm.T)
test_intense = np.diagonal(norm_freq)
print("intensity in unknown units: \n", test_intense)
test_intense_kmmol = test_intense*42.2561
print("intensity in kmmol: \n", test_intense_kmmol)

#################################### End of Normal mode testing code #############################

infile.close()

pq = np.dot(apt.T, modes_unweight)   #shape 3x3N
print(pq.T)
pq_pq = np.dot(pq.T, pq)    #shape 3Nx3N
intense = np.diagonal(pq_pq)
print("intensity in unknown units: \n", intense)
intense_kmmol = intense*42.2561
print("intensity in kmmol: \n", intense_kmmol)

np.set_printoptions(precision=7)
os.chdir('../../')
np.save('energy.npy', e)
np.save('gradient.npy', g)
np.save('hessian.npy', h)
np.save('apt.npy', apt)
print("MIM Energy:", e, "Hartree")
print("MIM Gradient:\n", g)
print("MIM Hessian shape:\n", h.shape, "\n")
print("MIM mass-weighted APT's shape:", apt.shape)

for i in range(0, len(freq)):
    print("Freq:", freq[i], "int :", intense_kmmol[i])

print("apt from atomic:\n", apt)
#print("\napt from grad:\n", apt_grad)
