#!/usr/bin/env python

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import traceback

from xcsvm.solvers import *
import xcsvm.utils.base as ubase
import xcsvm.utils.log as ulog
import xcsvm.utils.mpi as umpi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs xmcsvm solver.")
    parser.add_argument('solver_id', type=str,
                        help="Id of the solver.")
    parser.add_argument('--train_data', '-tr', type=str,
                        help="Path to train data.", default=None)
    parser.add_argument('--test_data', '-te', type=str,
                        help="Path to test data.", default=None)
    parser.add_argument('--test_output', '-to', type=str,
                        help="Path for test output.", default=None)
    parser.add_argument('--model_dir', '-m', type=str,
                        help="Path to model directory.", default=None)
    parser.add_argument('--options', '-O', type=str, default=[],
                        help="Solver options.", nargs="*")
    parser.add_argument('--verbosity', '-v', type=int, default=1,
                        help="Set verbosity.")
    parser.add_argument('--profile', '-P', dest='profile', action='store_true',
                        default=False,
                        help="Profile the training.")
    parser.add_argument('--line_profile', '-LP', dest='line_profile',
                        action='store_true', default=False,
                        help="Line profile the cython training.")

    args = parser.parse_args()
    if((args.train_data is None and args.model_dir is None) or
       (args.test_data is None and args.model_dir is None) or
       (args.train_data is None and args.test_data is None)):
        parser.error("At least two of train_data, test_data and"
                     " model_dir are required.")

    rank = umpi.MPIHelper().rank()
    log = ulog.MPILogger(args.verbosity, rank)

    if args.train_data is not None:
        options = {x.split("=")[0]: heuristic_conversion(x.split("=")[1])
                   for x in args.options if len(x.strip()) > 0}
        if args.profile is True:
            options["mode"] = "profiling"
        if args.line_profile is True:
            options["mode"] = "line_profiling"
        options["verbosity"] = args.verbosity

        class_ = SOLVERS[args.solver_id]
        solver = class_(**options)

        log.info("Loading train data.")
        X, y = get_data(args.train_data, solver.dtype)

        log.info("Start training.")

        def f():
            return solver.fit(X, y)
        _, msg = ubase.time_it(f, "Training",
                               profile=args.profile,
                               line_profile=args.line_profile)
        log.info(msg)
    else:
        log.info("Loading model.")

        def f():
            return BaseXMCSolver.unserialize(args.model_dir)
        solver, msg = ubase.time_it(f, "Unserializing")
        log.info(msg)

    if args.test_data is not None:
        try:
            log.info("Loading test data.")
            X, y = ubase.get_data(args.test_data, solver.dtype)

            log.info("Start prediction.")

            def f():
                return solver.predict(X)
            y_hat, msg = time_it(lambda: solver.predict(X), "Test")
            log.info(msg)

            if rank == 0:
                acc = (y == y_hat).mean()
                log.info("Accuracy: %f" % (acc,))

            if args.test_output is not None:
                with open(args.test_output, "w") as f:
                    f.write("\n".join(["%i" % x for x in y_hat]))

        except Exception as e:
            print "Error during prediction:"
            traceback.print_exc()

    if args.train_data is not None and args.model_dir is not None:
        # only store if we did some training
        log.info("Storing model.")

        def f():
            return solver.serialize(args.model_dir)
        _, msg = time_it(f, "Serializing")
        log.info(msg)
    pass
