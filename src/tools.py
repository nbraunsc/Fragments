def recurse(f_old, start, derivs):
    if fj > start:
        df_new = fj.intersection(f_old)
        df_new.coeff = f_old.coeff * -1
        
        if len(df_new) > 0:
            derivs.append(df_new)
            recurse(f_new, fj, derivs)

def run_pie(self, frag.frag): #principle of inclusion-exclusion
    derivs = []
    for fi in frag.frag:
        dfi = fi
        dfi.coeff = 1
        derivs.append(dfi)
        recurse(fi, fi, derivs)



