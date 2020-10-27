# Molecules-in-Molecules (MIM)
Fresh implementation of fragmetation code MIM in python using PySCF.  

Need to make sure all the requried packages are installed.  Can be run on both a cluster system or local laptop.  Can be computed in parallel for both instances. Locally there is the option of using Ray or sow/reap method, on a cluster there is only the sow/reap method.

Need to make sure the input file of desried molecule has bonding order and is in the inputs directory.

Code is able to do the following:
- MIM1 and MIM2 energies, gradients, and hessians
- Normal modes and IR intensities

## Getting the MIM code
### Using GitHub
Clone the public repository:

'git clone https://github.com/nbraunsc/Fragments'

Then use pip install to install the python package:

'pip install -e .'

## Running on cluster
### Sow/Reap method:
Need to specify parameters in mim1.py or mim2.py for specific calculaion. Then:

`qsub -N mim1 pbs_1.sh` or `qsub -N mim2 pbs_1.sh`

After pbs_1.sh is complete run the following bash script:

`./submit_mult.sh`

This will submit all individual fragment jobs to the cluster and queue the python script that recombines all subcalculations for global properties.

## Running on local machine
### Sow/Reap method:

Need to specify parameters in mim1.py or mim2.py for specific calculaion submit one of these two python commands depending on which method desired (MIM1 or MIM2):

`python mim1.py` or `python mim2.py`

This will start the fragmetation and create all input files for the subcalculations in "to_run" directory.

To submit subcalculations run the following python command:

`python submit_mult.py`

Finally, once all subcalculations are done, run following python command to recombine all the subcalculations:

`python global.py`

### Ray method:

Need to specify parameters at the bottom of MIM.py script in nicolefragment directory then:

`python MIM.py`




