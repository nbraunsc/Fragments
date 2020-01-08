

class Fragment():
    """
    Class to store a list of primitives corresponding to a molecular fragment
    """
    #def __init__(self, prims, molecule, attached=[], coeff=1):
    def __init__(self, prims, molecule, coeff=1):
        self.prims = prims 
        self.molecule = molecule 
        self.coeff = coeff 
        self.attached = []

    def __str__(self):
        out = "Frag:"
        out += str(self.prims)
        out += str(self.coeff)
        return out 

    def __repr__(self):
        return str(self)
