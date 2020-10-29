
#Input for MIM

$molecule
   0 3
   C     0.0000000    0.0000000    0.0000000
   H    -0.8611113    0.0000000    0.6986839
   H     0.8611113    0.0000000    0.6986839
$end

molecule_name = 'largermol'
mim_type = mim2            #mim1 or mim2
frag_type = 'distance'  #can be 'distance' or 'graphical'
frag_deg = 1.8  #smaller fragmentation level
frag_deg2 = 5  #larger fragmentation level
basis = 'sto-3g'
low_theory = 'RHF'
high_theory = 'MP2'
software = Pyscf.Pyscf          #could be Molcas.Molcas or Qchem.Qchem
stepsize = 0.001        #for second derivative by finite difference
