def recurse(f_old, start, derivs, fraglist):
    for fj in range(start+1, len(fraglist)):
        if fj > start:
            df_new = fraglist[fj].intersection(f_old)
            if len(df_new) > 0:
                derivs.append(df_new)
                print(derivs, 'derivs')
                recurse(df_new, fj, derivs, fraglist)

def runpie(fraglist):
    derivs = []
    for fi in range(0, len(fraglist)):
        dfi = fraglist[fi]
        derivs.append(dfi)
        recurse(derivs[fi], fi, derivs, fraglist)
    return derivs
