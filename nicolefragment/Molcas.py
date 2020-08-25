import numpy as np
import os
import sys

class Molcas():
    """ OpenMolcas quantum chemistry backend class.

    Will write the input file and submit it to run in OpenMolcas.  Will also use the .xyz file that was written in the Fragmentation() class.
    """

    def __init__(self):
        #self.inputxyz = inputxyz
        pass

    def write_xyzinput(self):
        f = open('fragment.input', 'w+')
        xyz = open("../inputs/water.xyz", "r")
        xyz_info = xyz.read()
        xyz.close()
        L = ["&GATEWAY \n", "coord= \n", xyz_info, "basis \n", 'sto-3g \n', "&SEWARD \n", "&SCF"]
        f.writelines(L)
        f.close()

    def energy_gradient(self):
        bash_command = "pymolcas fragment.input -f"
        os.system(bash_command)
        outfile = open('fragment.log', 'r')
        outfile_lines = outfile.readlines()
        for line in outfile_lines:
            if line.startswith("::    Total SCF energy"):
                print(line)
                #e = pull out energy
                print("energy = ", e)
        outfile.close()
        #return e, g

if __name__ == "__main__":
    x = Molcas()
    x.write_xyzinput()
    x.energy_gradient()
    

