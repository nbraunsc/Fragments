def recurse(f_old, start, derivs, fraglist):
    print(f_old, 'f_old')
    print(start, 'start')
    for fj in range(0, len(derivs)):
        if fj > start:
            print(fj, 'fj', start, 'start')
            df_new = derivs[fj].intersection(f_old)
            print(df_new, 'dfnew')
            #df_new_coeff = f_old_coeff * -1
            if len(df_new) > 0:
                derivs.append(df_new)
                recurse(df_new, fj, derivs, fraglist)

def run_pie(fraglist): #principle of inclusion-exclusion
    derivs = []
    for fi in fraglist:
        dfi = fi
        dfi_coeff = 1
        derivs = fraglist
        recurse(derivs[0], 0, derivs, fraglist)
    return derivs

### Also need to figure out how to operate a dictonary within Fragment class ###
