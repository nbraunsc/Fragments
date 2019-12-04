def recurse(f_old, start, derivs, fraglist):
    for fj in range(start+1, len(fraglist)):
        if fj > start:
            df_new = fraglist[fj].intersection(f_old)
            #df_new.coeff = f_old.coeff * -1
            if len(df_new) > 0:
                derivs.append(df_new)
                recurse(df_new, fj, derivs, fraglist)

def runpie(fraglist):
    derivs = []
    for fi in range(0, len(fraglist)):
        dfi = fraglist[fi]
        #dfi.coeff = 1
        derivs.append(dfi)
        recurse(derivs[fi], fi, derivs, fraglist)
    print(derivs)
    return derivs
