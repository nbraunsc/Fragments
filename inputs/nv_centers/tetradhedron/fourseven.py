import pyscf
from pyscf import molden
from pyscf import gto, dft, scf, ci, cc, mp, hessian, lib, grad, mcscf
from pyscf.geomopt.berny_solver import optimize

#DFT optimized geometry of C49H52N- tetrahedron
mol = gto.Mole()
mol.atom = '''
C   1.531421   3.106334   1.548830
C   0.631121   3.982995   0.664012
C   1.504999   4.878983  -0.235159
C   2.415646   5.778126   0.626989
C   3.319524   4.883151   1.514338
C   2.441911   3.981356   2.437495
C   3.346726   3.099429   3.290873
C   4.225402   3.974582   4.177876
C   3.318942   4.840662   5.103902
N   2.308049   5.598886   4.365187
C   1.558735   6.591557   5.137026
C   0.670576   7.510424   4.244793
C  -0.216694   6.643571   3.358190
C   0.653578   5.759918   2.471364
C  -0.233332   4.861317   1.582313
H  -0.877830   4.229700   2.213613
H  -0.900905   5.499670   0.984457
C   1.533077   6.659947   1.548254
C   2.325140   7.687440   2.303664
C   1.550385   8.436894   3.348511 
C   2.450369   9.337906   4.233684
C   1.556670  10.196905   5.152634
C   0.682460   9.294499   6.045055
C  -0.199125   8.396097   5.163214
H  -0.866773   9.008575   4.538945
H  -0.843848   7.762887   5.792699
C   1.582396   8.389675   6.901480
C   2.476013   7.503862   6.006314
C   3.380892   6.630391   6.868247
C   4.242548   5.746596   5.973132
C   5.149453   4.841341   6.834693
C   6.042629   3.962587   5.944778
C   5.132542   3.092732   5.063237
H   5.736788   2.440156   4.415471
H   4.514347   2.433002   5.692094
C   6.923659   4.858165   5.052158
C   6.043026   5.764137   4.166207
C   5.130109   4.876196   3.281093
H   5.817342   4.160082   2.754993
C   4.357102   5.666293   2.265215
C   5.127276   6.682352   1.479741
C   4.206048   7.578460   0.602523
C   3.322185   8.477938   1.513945
C   4.223669   9.368776   2.416580
C   5.144826   8.470919   3.292023
C   4.376720   7.679704   4.305212
C   3.354031   8.430163   5.108158
H   3.873951   9.118560   5.827653
C   5.147377   6.646266   5.074479
H   5.846564   7.156270   5.790654
C   6.030531   7.571259   2.382507
C   6.893056   8.477643   1.479510
C   5.991246   9.372527   0.605106
C   5.091178   8.484824  -0.278392
H   4.457156   9.114050  -0.922520
H   5.711035   7.866282  -0.946226
C   5.108656  10.252945   1.513203
H   4.474976  10.911520   0.898777
H   5.741084  10.906411   2.134348
H   6.617741  10.014256  -0.034349
H   7.542884   7.858972   0.840936
H   7.555193   9.101653   2.099985
C   6.902450   6.666736   3.266855
H   7.554941   6.043025   2.632438
H   7.567269   7.286594   3.892179
H   5.854425   9.174414   3.805474
C   3.331599  10.218984   3.334056
H   2.692883  10.879994   2.723856
H   3.959737  10.875184   3.960221
H   2.794065   9.186261   0.820074
C   3.296870   6.680923  -0.250832
H   2.657593   7.305918  -0.897334
H   3.912477   6.057609  -0.921473
H   5.824886   6.171415   0.762566
H   7.585693   5.477471   5.677071
H   7.573691   4.234946   4.418596
H   6.679167   3.320116   6.572981
H   4.531593   4.209925   7.492292
H   5.766060   5.474919   7.489652
H   2.772495   6.006039   7.544982
H   4.021970   7.262838   7.504204
H   0.966542   7.756327   7.559092
H   2.224986   8.997510   7.555860
H   0.048605   9.915767   6.696855
H   0.918470  10.855146   4.542903
H   2.184064  10.850656   5.778197
H   0.820900   9.129912   2.849084
H   0.801808   7.175451   0.869022
C   1.541834   4.847297   3.369865
H   0.845018   4.136768   3.884333
H  -0.880480   6.019309   3.980779
H  -0.865973   7.280764   2.735417
H   0.863730   6.074522   5.847481
H   2.819045   4.129530   5.810637
H   3.975575   2.465738   2.644037
H   2.737784   2.420395   3.912189
H   3.825720   4.167839   0.811530
H   0.866273   5.503180  -0.879168
H   2.120167   4.256236  -0.903074
H  -0.014803   3.345328   0.040514
H   2.161989   2.453990   0.926395
H   0.915433   2.446247   2.179471
'''

mol.basis = '6-31g*'
mol.charge = -1
mol.spin = 2 #nelec_alpha - nelec_beta
mol.build(symmetry=True)
mf = scf.RHF(mol)
mf.kernel()

## 6 orbitals, 6 electrons
#state_id : 0 is ground state, state_id=1 would be first
state_id = 0
mc = mcscf.CASSCF(mf, 7, 4).state_specific_(state_id)
mc.verbose = 5
hartree = mc.kernel()[0]
energy = hartree*27.2114
molden.from_mo(mol, 'four_ground.molden', mc.mo_coeff)
print("energy in eV = ", energy)

state_id = 1
mce = mcscf.CASSCF(mf, 7, 4).state_specific_(state_id)
mce.verbose = 5
hartreee = mce.kernel()[0]
energye = hartreee*27.2114
molden.from_mo(mol, 'four_excited.molden', mce.mo_coeff)
print("energy ground in eV = ", energy)
print("energy excited in eV = ", energye)
print("E diff in eV = ", abs(energye-energy))
