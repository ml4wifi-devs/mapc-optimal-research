from dataclasses import dataclass
from chex import Array, Scalar
from typing import Dict

import os
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

from mapc_research.plots import confidence_interval, set_style

RESULTS_PATH = "mapc_research/scalability/results"
DB_COLUMNS = ["start_timestamp", "n_sta_per_ap", "x_aps", "y_aps", "repetition", "seed", "time"]


def create_db(name: str) -> str:

    path = os.path.join(os.getcwd(), "mapc_research", "scalability", "database", name)
    path = path if path.endswith(".csv") else path + ".csv"

    if os.path.exists(path):
        return path
    
    db = open(path, "a")
    db.write(",".join(DB_COLUMNS) + "\n")
    db.close()

    return path

def plot_results(df: pd.DataFrame, save_path: str, log_space: bool = False):

    # Unpack results
    aps = df["n_aps"].unique()
    times_mean = df.groupby('n_aps').mean().reset_index()["time"].values
    times_ci_low = df.groupby('n_aps').apply(lambda x: confidence_interval(x, ci=0.95)[1])["time"].values
    times_ci_high = df.groupby('n_aps').apply(lambda x: confidence_interval(x, ci=0.95)[2])["time"].values

    # Fit curve
    a, b = jnp.polyfit(aps.astype(float), jnp.log(times_mean), 1)
    scale, exponent = jnp.exp(b), jnp.exp(a)

    # Define resolution for fitted curve
    xs = jnp.linspace(0, aps[-1], 100)

    # Plot results
    set_style()
    plt.figure(figsize=(4,3))
    plt.scatter(aps, times_mean, c="C0", label="Data", marker="x")
    plt.fill_between(aps, times_ci_low, times_ci_high, color="C0", alpha=0.3)
    plt.plot(
        xs, scale*jnp.power(exponent, xs),
        c="tab:grey", linestyle="--", linewidth=0.5, label=f"Fit"
    )
    plt.yscale("log" if log_space else "linear")
    plt.xlabel("Number of access points")
    plt.ylabel("Execution time [s]")
    plt.legend()
    plt.title(f"scale = {scale:.5f}, exponent = {exponent:.5f}")
    plt.tight_layout()
    plt.savefig(save_path)


@dataclass
class ExperimentResult:
    config: Dict
    aps: Array
    times_mean: Array
    times_std_low: Array
    times_std_high: Array
    total_time: Scalar
    scale: Scalar
    exponent: Scalar