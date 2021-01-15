import pickle
import os
import glob
import dill
import numpy as np

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
freq, modes = new_class.mw_hessian(h)
mole_coords = new_class.molecule.atomtable
labels = []
for i in range(0, len(mole_coords)):
    label = mole_coords[i][0]
    mole_coords[i].remove(label)
    labels.append(label)

atoms = np.array(labels)
print(atoms)
coords = np.array(mole_coords).flatten()
print(coords, coords.shape, "\n")
x = np.indices(modes.shape)[1]
apt_norm = np.zeros((3*new_class.molecule.natoms, 3))

def normal_disp(modes_list):
    for i in modes_list:
        print("mode:", i)
        positive_coords = coords + step*modes[i] #getting new global coords for displacement along normal mode
        pos = positive_coords.reshape((3,3))
        neg_coords = coords - step*modes[i] #getting new global coords for displacement along normal mode
        neg = neg_coords.reshape((3,3))
        
        all_data_pos = np.vstack((atoms, pos.T)).T
        all_data_neg = np.vstack((atoms, neg.T)).T
        
        new_class.molecule.atomtable = all_data_pos #setting positive disp global coords 
        inputxyz = new_class.build_xyz()    #getting pyscf xyz input format
        dip = new_class.qc_class.get_dipole(inputxyz)   #getting dipole with new coords
        new_class.molecule.atomtable = all_data_neg  #setting negative disp  global coords 
        inputxyz = new_class.build_xyz()    #getting pyscf xyz input format
        dip2 = new_class.qc_class.get_dipole(inputxyz)   #getting dipole with new coords
        apt_comp = (dip-dip2)/2*step    #3x1 vector
        print(apt_comp)
        apt_norm[i] = apt_comp

normal_disp(x[0])
print(apt_norm)
norm_freq = np.dot(apt_norm, apt_norm.T)
intense = np.diagonal(norm_freq)
print(intense)
intense_kmmol = intense*42.2561
print(intense_kmmol)
exit()
infile.close()
pq = np.dot(apt.T, modes)   #shape 3x3N
pq_pq = np.dot(pq.T, pq)    #shape 3Nx3N
intense = np.diagonal(pq_pq)
intense_kmmol = intense*42.2561

np.set_printoptions(suppress=True)
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
print("\napt from grad:\n", apt_grad)
