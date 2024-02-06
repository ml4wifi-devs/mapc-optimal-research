from typing import Dict

import os
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

from mapc_research.utils import *
from mapc_research.scalability import *
from mapc_research.plots import set_style, get_cmap

CMAP = get_cmap(2)
COLOR_MAP = {
    "PULP": CMAP[0],
    "COPT": CMAP[1],
}
STYLE_MAP = {
    "min_thr": "dashed",
    "total_sum": "dotted",
}


def plot_cumulative(results: Dict[str, ExperimentResult], title: str = None):

    # Set plot style
    set_style()
    
    # Create figure
    plt.figure(figsize=(4,3))
    
    # Iterate over results
    for key in sorted(results.keys()):
        result = results[key]

        # Get result specification
        solver = key.split("-")[0]
        opt_task = key.split("-")[-1]
        
        # Get exponential fit
        shift, exponent = result.shift, result.exponent

        # Generate data
        xs = jnp.linspace(1, result.aps[-1], 100)
        ys = shift + jnp.power(exponent, xs)

        # Plot data
        label = f"{solver}, {opt_task}, O({exponent:.3f}**n)"
        plt.plot(xs, ys, label=label, c=COLOR_MAP[solver], linestyle=STYLE_MAP[opt_task])
    
    plt.legend()
    plt.xlabel("Number of APs")
    plt.ylabel("Execution time [s]")
    # plt.yscale("log")
    plt.tight_layout()
    plt.savefig("cumulative_plot.pdf")

if __name__ == "__main__":
    
    # Load results
    results = os.listdir(RESULTS_PATH)
    results = [file for file in results if file.endswith(".pkl")]
    results = {file.split(".")[0]: load(os.path.join(RESULTS_PATH, file)) for file in results}

    # Plot cumulative time
    plot_cumulative(results, title="Scalability Experiments")
