import h5py
import numpy as np
import numba
import os
import argparse
from data import load_data

import inference
import post_process


def fit(
        in_file,
        out_file,
        convergence_threshold=1e-6,
        density="binomial",
        annealing_power=1.0,
        max_iters=int(1e4),
        mix_weight_prior=1.0,
        num_annealing_steps=1,
        num_clusters=10,
        num_grid_points=100,
        num_restarts=1,
        num_threads=1,
        precision=200,
        print_freq=100,
        seed=None,
):
    numba.set_num_threads(num_threads)

    if seed is not None:
        np.random.seed(seed)

    log_p_data, mutations, samples = load_data(
        in_file, density, num_grid_points, precision=precision
    )

    best_elbo = float("-inf")

    result = None

    for i in range(num_restarts):
        print("Performing restart {}".format(i))

        priors = inference.get_priors(num_clusters, num_grid_points)

        priors.pi = np.ones(num_clusters) * mix_weight_prior

        var_params = inference.get_variational_params(
            len(priors.pi),
            log_p_data.shape[0],
            log_p_data.shape[1],
            log_p_data.shape[2],
        )

        elbo_trace = inference.fit_annealed(
            log_p_data,
            priors,
            var_params,
            annealing_power=annealing_power,
            convergence_threshold=convergence_threshold,
            max_iters=max_iters,
            num_annealing_steps=num_annealing_steps,
            print_freq=print_freq,
        )

        if elbo_trace[-1] > best_elbo:
            best_elbo = elbo_trace[-1]

            result = (elbo_trace, var_params)

        print("Fitting completed")
        print("ELBO: {}".format(elbo_trace[-1]))
        print(
            "Number of clusters used: {}".format(len(set(var_params.z.argmax(axis=1))))
        )
        print()

    elbo_trace, var_params = result

    print("All restarts completed")
    print("Final ELBO: {}".format(elbo_trace[-1]))
    print("Number of clusters used: {}".format(len(set(var_params.z.argmax(axis=1)))))

    with h5py.File(out_file, "w") as fh:
        fh.create_dataset(
            "/data/mutations",
            data=np.array(mutations, dtype=h5py.string_dtype(encoding="utf-8")),
        )

        fh.create_dataset(
            "/data/samples",
            data=np.array(samples, dtype=h5py.string_dtype(encoding="utf-8")),
        )

        fh.create_dataset("/data/log_p", data=log_p_data)

        fh.create_dataset("/priors/pi", data=priors.pi)

        fh.create_dataset("/priors/theta", data=priors.theta)

        fh.create_dataset("/var_params/pi", data=var_params.pi)

        fh.create_dataset("/var_params/theta", data=var_params.theta)

        fh.create_dataset("/var_params/z", data=var_params.z)

        fh.create_dataset("/stats/elbo", data=np.array(elbo_trace))


def write_results_file(in_file, out_file, compress=False):
    df = post_process.load_results_df(in_file)

    df = post_process.fix_cluster_ids(df)

    if compress:
        df.to_csv(
            out_file, compression="gzip", float_format="%.4f", index=False, sep="\t"
        )

    else:
        df.to_csv(out_file, float_format="%.4f", index=False, sep="\t")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        help="Path to TSV format file with copy number and allele count information for all samples.")
    parser.add_argument("--output_file", help="Path to where results will be written in HDF5 format.")
    parser.add_argument("--num_clusters", type=int,
                        help="Number of clusters to use in variational approximation distribution. Note that not all clusters may not be assigned data points, so the final number of clusters could be lower. Default is 10.")
    parser.add_argument("--density", type=str,
                        help="Allele count density in the PyClone model. Use beta-binomial for high coverage sequencing.")
    parser.add_argument("--num_restarts", type=int,
                        help="Number of random restarts of variational inference. Default is 1.")
    parser.add_argument("--num_annealing_steps", type=int,
                        help="Number of simulated annealing steps to use. Default is one step i.e. not to use simulated annealing.")
    parser.add_argument("--num_grid_points", type=int,
                        help="Number of points used to approximate CCF values. Default is 100.")
    parser.add_argument("--annealing_power", type=float,
                        help="Exponent of entries in the annealing ladder . Default is 1.0.")
    parser.add_argument("--convergence_threshold", type=float,
                        help="Maximum relative ELBO difference between iterations to decide on convergence. Default is 10^-6.")
    parser.add_argument("--max_iters", type=int,
                        help="Maximum number of ELBO optimization iterations. Default is 10,0000.")
    parser.add_argument("--mix_weight_prior", type=float,
                        help="Parameter value of symmetric Dirichlet prior distribution on mixture weights. Higher values will produce more clusters. Default is 1.0 which is the uniform prior.")
    parser.add_argument("--precision", type=int,
                        help="Precision for Beta-Binomial density. Has no effect when using Binomial.")
    parser.add_argument("--seed", type=int,
                        help="Set random seed so results can be reproduced.")

    args = parser.parse_args()
    if args.input_file:
        input_file = args.input_file
    if args.output_file:
        output_file = args.output_file
    if args.num_clusters:
        num_clusters = args.num_clusters
    if args.density:
        density = args.density
    if args.num_restarts:
        num_restarts = args.num_restarts
    if args.num_annealing_steps:
        num_annealing_steps = args.num_annealing_steps
    if args.num_grid_points:
        num_grid_points = args.num_grid_points
    if args.annealing_power:
        annealing_power = args.annealing_power
    if args.convergence_threshold:
        convergence_threshold = args.convergence_threshold
    if args.max_iters:
        max_iters = args.max_iters
    if args.mix_weight_prior:
        mix_weight_prior = args.mix_weight_prior
    if args.precision:
        precision = args.precision
    if args.seed:
        seed = args.seed

    tmp_output = os.path.splitext(os.path.basename(input_file))[0] + ".h5"
    fit(input_file,
        tmp_output,
        num_clusters=num_clusters,
        density=density,
        num_restarts=num_restarts,
        num_annealing_steps=num_annealing_steps,
        num_grid_points=num_grid_points,
        annealing_power=annealing_power,
        convergence_threshold=convergence_threshold,
        max_iters=max_iters,
        mix_weight_prior=mix_weight_prior,
        precision=precision,
        seed=seed
        )
    write_results_file(tmp_output, output_file)
