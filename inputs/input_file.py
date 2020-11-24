#Input for MIM
import sys

#number of mim levels
mim_levels = 2            

#can be 'distance' or 'graphical'
frag_type = 'distance' 

#smaller fragmentation level
frag_deg = 1.6

#larger fragmentation level
frag_deg_large = 3

#Basis set for quantum calculation
basis_set = 'sto-3g'

#Always need to define high_theory
high_theory = 'MP2'

#Only define low_theory if mim_levels = 2
low_theory = 'RHF'

#could be Molcas.Molcas or Qchem.Qchem
software = 'Pyscf'  

#for second derivative by finite difference
stepsize = 0.001        

#batch_size for running calculations
batch_size = 3

#geometry optimization set to True or False
opt = False
#opt = True
