import os
os.environ['JAX_ENABLE_X64'] = "True"

import traceback
import time
import json
from itertools import chain
from tqdm import tqdm
from typing import Dict

import jax
import jax.numpy as jnp
from scipy.optimize import curve_fit
from chex import PRNGKey
import matplotlib.pyplot as plt
from mapc_optimal import Solver, positions_to_path_loss
from argparse import ArgumentParser

from mapc_research.envs.static_scenarios import *
from mapc_research.plots import confidence_interval, set_style
from mapc_research.utils import *
from mapc_research.scalability import *


def measure_point(
        n_reps: int,
        scenario_config: dict,
        key: PRNGKey,
        verbose: bool = False,
        solver_kwargs: dict = None
    ):

    # Draw integer seed for each repetition
    seeds = jax.random.randint(key, shape=(n_reps,), minval=0, maxval=jnp.iinfo(jnp.int32).max)

    # Iterate over repetitions
    times = []
    for i in range(n_reps):
        # Draw seed for this repetition
        scenario_config["seed"] = seeds[i]

        # Create scenario
        scenario = random_scenario(**scenario_config)
        if verbose:
            scenario.plot()

        # Get associations
        associations = scenario.get_associations()
        access_points = list(associations.keys())
        stations = list(chain.from_iterable(associations.values()))

        # Get path loss
        positions = scenario.pos
        walls = scenario.walls
        path_loss = positions_to_path_loss(positions, walls)

        # Define solver
        solver = Solver(stations, access_points, **(solver_kwargs or {}))

        # Solve and measure time
        try:
            start = time.time()
            configurations, rate, objectives = solver(path_loss, return_objectives=True)
            times.append(time.time() - start)
        except Exception as e:
            traceback.print_exc()
            times.append(jnp.nan)

    return jnp.asarray(times)


def save_checkpoint(aps: jnp.ndarray):

    # Fit exponential curve to times
    if len(aps) < 3:
        shift, exponent = jnp.nan, jnp.nan
    else:
        (shift, exponent), _ = curve_fit(lambda x, s, e: s + jnp.power(e, x), aps, times_mean)

    # Save results
    result = ExperimentResult(
        config=experiment_config,
        aps=aps,
        times_mean=jnp.asarray(times_mean),
        times_std_low=jnp.asarray(times_std_low),
        times_std_high=jnp.asarray(times_std_high),
        total_time=total_time,
        shift=shift,
        exponent=exponent
    )
    save(result, os.path.join(
        RESULTS_PATH,
        f"{args.solver.upper()}-max_aps{max_aps}-n_reps{experiment_config['n_reps']}-{opt_task}.pkl"
    ))

    # Plot results
    plot_results(result)

def plot_results(experiment_results: ExperimentResult):

    # Unpack results
    aps = experiment_results.aps
    times_mean = experiment_results.times_mean
    times_std_low = experiment_results.times_std_low
    times_std_high = experiment_results.times_std_high
    total_time = experiment_results.total_time
    shift = experiment_results.shift
    exponent = experiment_results.exponent

    # Define resolution for fitted curve
    xs = jnp.linspace(aps[0], aps[-1], 100)

    # Plot results
    set_style()
    plt.figure(figsize=(4,3))
    plt.scatter(aps, times_mean, c="C0", label="Data", marker="x")
    plt.fill_between(aps, times_std_low, times_std_high, color="C0", alpha=0.3)
    plt.plot(
        xs, shift + jnp.power(exponent, xs),
        c="tab:grey", linestyle="--", linewidth=0.5, label=f"Fit"
    )
    plt.yscale("log" if args.log_space else "linear")
    plt.xlabel("Number of access points")
    plt.ylabel("Execution time [s]")
    plt.legend()
    plt.title(f"Total execution time: {total_time:.2f}s\nshift = {shift:.5f}, exponent = {exponent:.5f}")
    plt.tight_layout()
    plt.savefig(os.path.join(
        RESULTS_PATH,
        f"{args.solver.upper()}-max_aps{max_aps}-n_reps{experiment_config['n_reps']}-{opt_task}.pdf"
    ))


if __name__ == "__main__":

    # Load experiment configuration from config file
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-s", "--solver", type=str, default="pulp", choices=["pulp", "copt", "scip", "glpk", "choco"])
    parser.add_argument("-l", "--log-space", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    experiment_config = json.load(open(args.config, "r"))

    # Define range of access points
    min_aps = experiment_config["min_aps"]
    max_aps = experiment_config["max_aps"]
    step_aps = experiment_config["step_aps"]
    if args.log_space:
        aps = jnp.logspace(jnp.log10(min_aps), jnp.log10(max_aps), num=(max_aps-min_aps)//step_aps+1, dtype=int)
    else:
        aps = jnp.linspace(min_aps, max_aps, num=(max_aps-min_aps)//step_aps+1, dtype=int)

    # Define solver kwargs
    solver_kwargs = {
        "solver": SOLVERS[args.solver](msg=args.verbose),
        "opt_sum": experiment_config["opt_sum"],
    }
    
    # Iterate over access points and measure exec times
    opt_task = "total_sum" if experiment_config["opt_sum"] else "min_thr"
    time_start = time.time()
    times_mean = []
    times_std_low = []
    times_std_high = []
    for i, n_ap in enumerate(tqdm(aps), start=1):
        # Define scenario config
        scenario_config = {
            "n_ap": n_ap,
            "d_sta": experiment_config["d_sta"],
            "n_sta_per_ap": experiment_config["n_sta_per_ap"],
            "ap_density": experiment_config["ap_density"]
        }

        # Measure exec time
        times = measure_point(
            n_reps=experiment_config["n_reps"],
            scenario_config=scenario_config,
            key=jax.random.PRNGKey(experiment_config["seed"]),
            verbose=args.verbose,
            solver_kwargs=solver_kwargs
        )
        mean, ci_low, ci_high = confidence_interval(times)
        times_mean.append(mean)
        times_std_low.append(ci_low)
        times_std_high.append(ci_high)
        total_time = time.time() - time_start
        save_checkpoint(aps[:i])

    # Print total execution time
    print(f"Total execution time: {total_time:.2f}s")
