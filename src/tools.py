def recurse(f_old, start, derivs):
    for fj in derivs > start: #### This is the problem ####
        df_new = fj.intersection(f_old)
        df_new_coeff = f_old_coeff * -1
        if len(df_new) > 0:
            derivs.append(df_new)
            recurse(f_new, fj, derivs)

def run_pie(fraglist): #principle of inclusion-exclusion
    derivs = []
    for fi in fraglist:
        dfi = fi
        dfi_coeff = 1
        derivs.append(dfi)
        recurse(dfi, dfi, derivs)
    return derivs

### Also need to figure out how to operate a dictonary within Fragment class ###
