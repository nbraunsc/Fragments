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
An input file will need to be edited or created with calculation parameters. Then the "sow" set will be initated as follows:

`python sow.py <input_file> <coords_file>`

An example with an input file named `dummie_input.py` and a coordinates file named `largermol.cml` is as follows:

`python sow.py dummie_input.py largermol.cml`

#### Run
User has the ability to run batches of calculations using the following command:

`python batch.py <batch size>`

The `<batch size>` is an integer that the user defines to determine the batch size for how  many fragment calculations would like to be run at a time. The `<batch size>` may also be empty if all jobs want to be submitted at once.

#### Reap
The final step is the "reap" set and is run once all the fragment calculations are complete using the following command:

`python reap.py >> <outfile>`

Where `<outfile>` is the file where the output from the MIM calculation will be written.

## How to run MIM code (using Ray formalism)

Need to specify parameters at the bottom of MIM.py script in the source directory "nicolefragment", then:

`python MIM.py`




