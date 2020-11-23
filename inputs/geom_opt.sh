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
MKL_NUM_THREADS=1

source activate pyconda

cd $PBS_O_WORKDIR

finished='false'

while [ "$finished" != "true" ]
do
    finished=$(python checker.py)
    sleep 10
done

echo $finished
echo "Optimizer is done!"

while [ "$finished" == "true" ]
do
    python reap.py >> reap.out
done

echo "Reap is done!"

exit;
