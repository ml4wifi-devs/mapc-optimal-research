from dataclasses import dataclass
from chex import Array, Scalar
from typing import Dict

import os

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