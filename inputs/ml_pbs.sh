#PBS -l walltime=00:10:00:00
#PBS -l nodes=1:ppn=24
#PBS -l mem=20GB
#PBS -n
#PBS -A mlspring20
#PBS -W group_list=newriver

module purge
module load gcc/5.2.0
module load Anaconda/5.2.0


#need num of threads for python jobs, keep one until parallizing
$MKL_NUM_THREADS = 1

cd $PBS_O_WORKDIR
source activate pyconda
cd ../
python -m pip install -e .

cd $PBS_O_WORKDIR

FILE=drugs_ml

# every so often, copy the output file back here!!
#touch ./$FILE.out
#while true
#do
#   cp ./$FILE.out $PBS_O_WORKDIR/$FILE.running.out
#   sleep 60
#done&

# run python job

python $FILE.py  >> $FILE.out
# copy data back
#cp ./$FILE.out $PBS_O_WORKDIR/$FILE.out
exit;
