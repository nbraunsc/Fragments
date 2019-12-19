

class Fragment():
    """
    Class to store a list of primitives corresponding to a molecular fragment
    """
    def __init__(self, prims, molecule, attached=[], coeff=1):
        self.prims = prims 
        self.molecule = molecule 
        self.coeff = coeff 
    
    def __str__(self):
        out = "Frag:"
        out += str(self.prims)
        return out 

    def __repr__(self):
        return str(self)
