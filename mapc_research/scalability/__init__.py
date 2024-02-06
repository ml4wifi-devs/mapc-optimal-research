from dataclasses import dataclass
from chex import Array, Scalar
from typing import Dict

RESULTS_PATH = "mapc_research/scalability/results"


@dataclass
class ExperimentResult:
    config: Dict
    aps: Array
    times_mean: Array
    times_std_low: Array
    times_std_high: Array
    total_time: Scalar
    shift: Scalar
    exponent: Scalar