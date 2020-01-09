from pyscf import gto, scf

def do_pyscf(input_xyz):
    mol = gto.Mole()
    #mol.atom =  '/home/nbraunsc/Documents/Projects/MIM/myoutfile.txt'   #different atoms are seperated by ; or a line break
    mol.atom = input_xyz
    mol.basis = 'sto-3g'

    #mol.symmetry = 1
    #mol.charge = 1
    #mol.spin = 2   #This is 2S, difference between number of alpha and beta electrons
    #mol.verbose = 5    #this sets the print level globally 0-9
    #mol.output = 'path/to/my_log.txt'  #this writes the ouput messages to certain place
    #mol.max_memory = 1000 #MB  #defaul size can be defined withshell environment variable PYSCF_MAX_MEMORY
        #can also set memory from command line:
        #python example.py -o /path/to/my_log.txt -m 1000
    #mol.output = 'output_log'
    mol.build()
    m = scf.RHF(mol)
    energy = m.kernel()
    print('\n', mol.atom)
    return energy
