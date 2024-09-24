import os
os.environ['JAX_ENABLE_X64'] = "True"

import traceback
import time
import json
from itertools import chain
from tqdm import tqdm
from typing import Dict, List

import jax
import jax.numpy as jnp
from chex import PRNGKey
from mapc_optimal import Solver, positions_to_path_loss
from argparse import ArgumentParser
from datetime import datetime

from mapc_research.envs.test_scenarios import *
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
        scenario = residential_scenario(**scenario_config)
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

        # Update database
        update_database(
            db_path,
            time_start,
            n_sta_per_ap,
            x_aps,
            y_aps,
            i+1,
            seeds[i],
            times[i]
        )


def update_database(
        db_path: str,
        time_start: float,
        n_sta_per_ap: int,
        x_aps: int,
        y_aps: int,
        repetition: int,
        seed: int,
        time: float,
        verbose: bool = False
    ):
    
    start_timestamp = datetime.fromtimestamp(time_start).strftime("%Y-%m-%d %H:%M:%S")
    row = f"{start_timestamp},{n_sta_per_ap},{x_aps},{y_aps},{repetition},{seed},{time}\n"
    db = open(db_path, "a")
    db.write(row)
    db.close()
    print(f"[DB UPDATE] {row}") if verbose else None



if __name__ == "__main__":

    # Load experiment configuration from config file
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-s", "--solver", type=str, default="pulp", choices=SOLVERS.keys())
    parser.add_argument("-d", "--database_name", type=str, default="scalability")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    experiment_config = json.load(open(args.config, "r"))

    # Define range of access points
    aps = jnp.array([x * y for x, y in zip(experiment_config["x_apartments"], experiment_config["y_apartments"])])

    # Define solver kwargs
    solver_kwargs = {
        "solver": SOLVERS[args.solver](msg=args.verbose, mip=True),
        "opt_sum": experiment_config["opt_sum"],
    }

    # Create database
    db_path = create_db(args.database_name)
    
    # Iterate over access points and measure exec times
    opt_task = "total_sum" if experiment_config["opt_sum"] else "min_thr"
    n_sta_per_ap = experiment_config["n_sta_per_ap"]
    time_start = time.time()
    for i, n_ap in enumerate(tqdm(aps), start=1):
        x_aps = experiment_config["x_apartments"][i - 1]
        y_aps = experiment_config["y_apartments"][i - 1]

        # Define scenario config
        scenario_config = {
            "x_apartments": x_aps,
            "y_apartments": y_aps,
            "n_sta_per_ap": n_sta_per_ap,
            "size": experiment_config["size"],
        }

        # Measure exec time
        measure_point(
            n_reps=experiment_config["n_reps"],
            scenario_config=scenario_config,
            key=jax.random.PRNGKey(experiment_config["seed"]),
            verbose=args.verbose,
            solver_kwargs=solver_kwargs
        )

    # Print total execution time
    print(f"Total execution time: {(time.time() - time_start):.2f}s")