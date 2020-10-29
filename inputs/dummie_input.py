#Input for MIM

$molecule
<file.cml> or manually input file.cml here
$end

molecule_name = 'largermol'
mim_type = mim1 or mim2
frag_type = 'distance'  #can be 'distance' or 'graphical'
frag_deg = 1.8  #smaller fragmentation level
frag_deg2 = 5  #larger fragmentation level
basis = 'sto-3g'
low_theory = 'RHF'
high_theory = 'MP2'
software = Pyscf.Pyscf
stepsize = 0.001        #for second derivative by finite difference
