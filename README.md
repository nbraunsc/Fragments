# Molecules-in-Molecules (MIM)
Fresh implementation of fragmetation code MIM in python using PySCF.  

Need to make sure all the requried packages are installed.  Can be run on both a cluster system or local laptop.  Can be computed in parallel for both instances. Locally there is the option of using Ray or sow/reap method, on a cluster there is only the sow/reap method.

Need to make sure the input file of desried molecule has bonding order and is in the inputs directory.

Code is able to do the following:
- MIM1 and MIM2 energies, gradients, and hessians
- Normal modes and IR intensities

## Getting the MIM code
#### Using GitHub
Clone the public repository:

`git clone https://github.com/nbraunsc/Fragments`

#### Using as a python module
The MIM code can be imported as a python package:

`import nicolefragment`

## How to run MIM code (using Sow/Reap formalism)
#### Sow
An input file will need to be edited with calculation parameters. Then "sow" step takes in two arguments, the input file and coordinate file:

`python sow.py <input_file> <coords_file>`

An example with an input file named `input_file.py` and a coordinates file named `water.cml` is as follows:

`python sow.py input_file.py water.cml`

#### Reap
The final step is the "reap" step and is run once all the fragment calculations are complete using the following command:

`python reap.py >> <outfile>`

Where `<outfile>` is the file where the output from the MIM calculation will be written.

