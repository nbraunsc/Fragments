

cd to_run
for i in *.sh; do
    echo "submitting $i"
    qsub $i
done 
