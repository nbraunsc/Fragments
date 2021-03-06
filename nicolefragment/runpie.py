def recurse(f_old, start, derivs, fraglist, signlist, sign):    #1st depth and on, finding intersection
    """ Recursive part of the pie
    
    Where funciton  builds layers and layers of fragments from interestions

    Parameters
    ----------
    f_old : list
        A list of primiatives for a specific fragment
    start : int
        The depth of intersections.  Always increasing.
    derivs : list
        List of derivates that keeps getting appended tol.l
    fraglist : list of lists
        List of fragment that gets generated by functions in the Fragmentation class. Does not change in this function.
    signlist : list
        List of coefficents
    sign : int
        New sign that is determined based on PIE
        
    """
    
    for fj in range(start, len(fraglist)):
        if fj > start:
            df_new = fraglist[fj].intersection(f_old)
            if len(df_new) > 0:
                derivs.append(df_new)
                df_newcoeff = sign * -1
                signlist.append(df_newcoeff)
                recurse(df_new, fj, derivs, fraglist, signlist, df_newcoeff)

def runpie(fraglist):
    """ Runs the principle of inculsion-exculsion
    
    Parameters
    ----------
    fraglist : list of lists
        List of fragments that gets generated by functions in the Fragmentation class.

    Returns
    -------
    derivs : list
        List of derivatives created by prinicple on inclusion-exculusion
    signlist : list
        List of coefficents (1 or -1)
    
    """
    
    derivs = []
    signlist = []
    for fi in range(0, len(fraglist)):  #0th depth, just initial frags
        dfi = fraglist[fi]
        dfi_coeff = 1
        derivs.append(dfi)
        signlist.append(dfi_coeff)
        recurse(fraglist[fi], fi, derivs, fraglist, signlist, dfi_coeff)
    return derivs, signlist
