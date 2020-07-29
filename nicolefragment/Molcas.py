import numpy as np
import os
import sys

class Molcas():
    """ OpenMolcas quantum chemistry backend class.

    Will write the input file and submit it to run in OpenMolcas.  Will also use the .xyz file that was written in the Fragmentation() class.
    """

    def __init__(self, inputxyz):
        self.inputxyz = inputxyz #input file in a string

    def write_xyzinput(self):
        pass
        #write molcas input with inputxyz

    def energy_gradient(self):
        pass
        #bash_command = pymolcas <name_of_inputfile> -f
        #os.system(bash_command)
        #outfile = open('<name_of_output_file>', 'r')
        #outfile_lines = outfile.read_lines()
        #outfile.close()
        #for line in outfile_lines:
        #    if line.startswith('Total SCF Energy'):
        #        e = pull out energy 
        #return e, g



