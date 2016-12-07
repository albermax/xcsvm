# xcsvm
Extreme Classification SVMs

This repository contains the code belonging to the following paper:

[Distributed Optimization of Multi-Class SVMs](http://arxiv.org/abs/1611.08480)
by *Maximilian Alber, Julian Zimmert, Urun Dogan, Marius Kloft*

**Please cite paper, if you make use of the code!**

## How to run this code

The file run.py is an interface for the code.
If you want to write your own scripts, please
consider this as an entry point.

```bash
usage: run.py [-h] [--train_data TRAIN_DATA] [--test_data TEST_DATA]
              [--test_output TEST_OUTPUT] [--model_dir MODEL_DIR]
              [--options [OPTIONS [OPTIONS ...]]] [--verbosity VERBOSITY]
              [--profile] [--line_profile]
              solver_id

Runs xmcsvm solver.

positional arguments:
  solver_id             Id of the solver.

optional arguments:
  -h, --help            show this help message and exit
  --train_data TRAIN_DATA, -tr TRAIN_DATA
                        Path to train data.
  --test_data TEST_DATA, -te TEST_DATA
                        Path to test data.
  --test_output TEST_OUTPUT, -to TEST_OUTPUT
                        Path for test output.
  --model_dir MODEL_DIR, -m MODEL_DIR
                        Path to model directory.
  --options [OPTIONS [OPTIONS ...]], -O [OPTIONS [OPTIONS ...]]
                        Solver options.
  --verbosity VERBOSITY, -v VERBOSITY
                        Set verbosity.
```

* Provide a training set via train_data to train or a model via model_dir to test.
* In case you provide a test set, the trained or loaded model will be used to test it.
* If you provide test_output the predicted labels will be stored in this file.
* **solver_id:** llw\_mr\_sparse for LLW and ww_sparse for WW.
* **Options:** Depending on the used solver you can set training options listed below.
* In case you want to use MPI, please launch the solver with mpiexec.

## Solver options

### All Solvers

* max_iter: maximum number of iterations over the training set. (default: 1000)
* C: the regularization parameter. (default: 1)
* epsilon: how precise to solve the problem. (default: 0.1)
* shuffle: shuffle the training set at each iteration. (default: True)
* seed: seed for the random number generator. (default: 1)
* dtype: data type for floating point values. (default: "float64") ["float32" | "float64"]
* idtype: data type for indices. (default: "uint64") ["uint32" | "uint64"]
* nr_threads: number of threads to use. (default: 1)

**llw**

* folds: split training set into folds and iterate over each as it would be a training set on its own. (default: 1)
* shrinking: enable shrinking. (default: 0) [0 | 1]
* shrink_state: shrink a sample/class combination after it is not updated for this number of iterations. (default: 1)

**ww**

* group\_count: join classes to group\_count number of groups. (default: 2*(nr\_threads*mpi\_processes))
* folds: split training set into folds and iterate over each as it would be a training set on its own. (default: 1)
* shrinking: enable shrinking. (default: 0) [0 | 1]
* shrink_state: shrink a sample/class combination after it is not updated for this number of iterations. (default: 1)

There are some more options available, though the are mainly for developing purposes.

## Disclaimer

The code is released under the MIT license.
