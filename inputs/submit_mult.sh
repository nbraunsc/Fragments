#!/bin/bash

#one=$(qsub pbs_1.sh)
#echo $one

#cd to_run/
#name=pbs_2.sh
#two=$(./$name)
#echo $two


cd to_run/
for f in *.sh; do
    two=$(qsub $f)
    echo "Submitting $f"
    echo $two
    echo ""
done

cd ../
name=final_pbs.sh
three=$(qsub -W depend=afterany:$two $name)
echo "Submitting global script, waits until fragments run"
echo $three
