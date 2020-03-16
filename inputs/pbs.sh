#PBS -l walltime=00:1:00:00
#PBS -l nodes=1:ppn=1
#PBS -l mem=10GB
#PBS -q nmayhall_lab
#PBS -A qcvt_doe
#PBS -W group_list=nmayhall_lab

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

FILE=asp_benz

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
cp ./$FILE.out $PBS_O_WORKDIR/$FILE.out
exit;
