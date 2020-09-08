#!/bin/bash

#one=$(qsub pbs.sh)
#echo $one

#cd to_run/
#name=second.sh
#two=$(./ -W depend=afteray:$one $name)
#echo $two

cd to_run/
for f in *.sh; do
    #if [[ $f == *.sh ]]
    #then
    two=$(qsub $f)
    #two=$(qsub -W depend=afterany:$one $f)
    echo $two
    #fi
done;
