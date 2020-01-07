def recurse(f_old, start, derivs, fraglist, signlist, sign):    #1st depth and on, finding intersection
    for fj in range(start, len(fraglist)):
        if fj > start:
            df_new = fraglist[fj].intersection(f_old)
            if len(df_new) > 0:
                derivs.append(df_new)
                df_newcoeff = sign * -1
                signlist.append(df_newcoeff)
                recurse(df_new, fj, derivs, fraglist, signlist, df_newcoeff)

def runpie(fraglist):
    derivs = []
    signlist = []
    for fi in range(0, len(fraglist)):  #0th depth, just initial frags
        dfi = fraglist[fi]
        dfi_coeff = 1
        derivs.append(dfi)
        signlist.append(dfi_coeff)
        recurse(fraglist[fi], fi, derivs, fraglist, signlist, dfi_coeff)
    return derivs, signlist
