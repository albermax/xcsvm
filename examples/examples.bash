#! /bin/bash

# The following script will show in a basic example
# how one can use this library.

# The ipython notebook script in this folder contains the same examples including the outputs!

# To compare the timing of the different runs one needs to wait until the cpu is cool again.

## Definitions #################################################################
TRAIN_DATA="../xcsvm/xcsvm/tests/data/lshtc1_dryrun_task1_tfidf_l2_norm_applied.train.libsvm"
TEST_DATA="../xcsvm/xcsvm/tests/data/lshtc1_dryrun_task1_tfidf_l2_norm_applied.test.libsvm"

# The following python interpreter should have xcsvm installed.
# If not used the run.py script will try to run the source code directly.
# In this case all the needed packages need to be installed.
PYTHON="/usr/bin/python2.7"


## Training and storing the model ##############################################
# Training 
rm -rf ./model
$PYTHON ../scripts/run.py ww_sparse --train_data $TRAIN_DATA --model_dir ./model --options folds=5 group_count=64 -v 1
# Testing
$PYTHON ../scripts/run.py ww_sparse --test_data $TEST_DATA --model_dir ./model


## Training and testing in one shot ############################################
sleep 60
$PYTHON ../scripts/run.py ww_sparse --train_data $TRAIN_DATA --test_data $TEST_DATA --options folds=5 group_count=64 -v 1


## Training with 2 threads #####################################################
sleep 60
$PYTHON ../scripts/run.py ww_sparse --train_data $TRAIN_DATA --test_data $TEST_DATA --options nr_threads=2 folds=5 group_count=64 -v 1


## Training with 2 processes ###################################################
# Requires openmpi.
sleep 60
mpiexec -n 2 $PYTHON ../scripts/run.py ww_sparse --train_data $TRAIN_DATA --test_data $TEST_DATA --options folds=5 group_count=64 -v 1
