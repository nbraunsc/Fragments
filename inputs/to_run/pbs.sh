#PBS -l walltime=00:1:00:00
#PBS -l nodes=2:ppn=4
#PBS -l mem=20GB
#PBS -q nmayhall_lab
#PBS -A qcvt_doe
#PBS -W group_list=nmayhall_lab

module purge
module load gcc/5.2.0
module load Anaconda/5.2.0


#need num of threads for python jobs, keep one until parallizing on single node
$MKL_NUM_THREADS = 1

cd $PBS_O_WORKDIR
source activate pyconda
cd ../../
python -m pip install -e .

cd $PBS_O_WORKDIR

#Display the contents of PBS_NODEFILE
echo ""
echo "Contents of PBS_NODEFILE:"
echo ""
cat $PBS_NODEFILE

#Display unique elements in PBS_NODEFILE
echo ""
echo "Unique node names:"
echo ""
uniq $PBS_NODEFILE

echo "file location test"
echo ""

FILE=first

# every so often, copy the output file back here!!
#touch ./$FILE.out
#while true
#do
#   cp ./$FILE.out $PBS_O_WORKDIR/$FILE.running.out
#   sleep 60
#done&

# run python job
python $FILE.py >> $FILE.out
#python $FILE.py `cat $PBS_NODEFILE | uniq` >> $FILE.out
# copy data back
#cp ./$FILE.out $PBS_O_WORKDIR/$FILE.out
exit;
