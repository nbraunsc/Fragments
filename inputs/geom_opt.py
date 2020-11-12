import pickle
import os
import glob

os.path.abspath(os.curdir)
os.chdir('to_run/')

def opt_fnc(coords):
    for i in os.listdir():
        os.chdir(i)
        files = glob.glob('*.status')
        for j in files:
            infile = open(j, 'rb')
            status = pickle.load(infile)
            if status == 0:
                sleep
            if status == -1:
                sleep
    return energy, gradient, hess, apt

if opt==True:
    frag.write_xyz(str(Molecule))
    os.path.abspath(os.curdir)
    os.chdir('../inputs/')
    optimizer = Berny(geomlib.readfile(os.path.abspath(os.curdir) + '/' + str(Molecule) + '.xyz'), debug=True)
    x = 0
    etot_opt = 0
    grad_opt = 0
    for geom in optimizer:
        x = x+1
        print("\n############# opt cycle:", x, "##################\n")
        solver = opt_fnc(geom.coords)
        optimizer.send(solver)
        etot_opt = solver[0]
        grad_opt = solver[1]
    relaxed = geom
    print("\n", "##########################", '\n', "#       Converged!       #", '\n', "##########################") 
    print('\n', "Energy = ", etot_opt)
    print('\n', "Converged_Gradient:", "\n", grad_opt)

else:
    return energy, gradient, hess, apt
