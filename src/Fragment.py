

class Fragment():
    """
    Class to store a list of primitives corresponding to a molecular fragment
    """
    def __init__(self, prims=[], coeff=1):
        self.prims = prims
        self.coeff = coeff 
    
    def __str__(self):
        out = "I'm a frag:"
        out += str(self.prims)
        return out 
