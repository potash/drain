#!/bin/bash
# Usage: run_models.sh <datadir> <outputdir> [paramsfile]
basename=$(dirname $0)
mkdir -p $2

# if specified, move the params file to the subdir
if [ -n "$3" ]; then
    cp $3 $2/params.yaml 
fi

# print number of models to run
python $basename/n_models.py $2/params.yaml

# delete old model runs
rm -r $2/*/ 2> /dev/null

python $basename/get_params.py $1 $2 $2/params.yaml | parallel -j1 --ungroup --delay 5 --joblog $2/log python $basename/run_model.py
