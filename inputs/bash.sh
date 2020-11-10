#!/bin/bash

cd to_run/

for j in *; do
    cd $j
    echo $j
    count=0
    declare -a my_array
    my_array=()
    for i in *.pickle; do
        my_array+=($i)
    done
    x=${#my_array[@]}
    cd ../
done
exit;
