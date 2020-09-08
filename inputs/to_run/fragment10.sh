#PBS -l nodes=2:ppn=4
#PBS -l mem=20GB
#PBS -q nmayhall_lab
#PBS -A qcvt_doe
#PBS -W group_list=nmayhall_lab
 
module purge
module load gcc/5.2.0
module load Anaconda/5.2.0
$MKL_NUM_THREADS = 1
cd $PBS_O_WORKDIR
source activate pyconda
cd ../../
python -m pip install -e . 
cd $PBS_O_WORKDIR
FILE=fragment10
python $FILE.py >> $FILE.out
exit;
