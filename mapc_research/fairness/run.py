import json
from collections import defaultdict
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
from mapc_optimal import Solver, positions_to_path_loss
from tqdm import tqdm

from mapc_research.envs.static_scenarios import simple_scenario_5, StaticScenario
from mapc_research.plots import get_cmap, set_style


def run_solver(scenario: StaticScenario, solver_kwargs: dict) -> tuple:
    associations = scenario.get_associations()
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))

    positions = scenario.pos
    walls = scenario.walls
    path_loss = positions_to_path_loss(positions, walls)

    solver = Solver(stations, access_points, **solver_kwargs)
    configurations, rate, objectives = solver(path_loss, return_objectives=True)

    return configurations, rate


def jains_fairness_index(data_rate: np.ndarray) -> float:
    n = len(data_rate)
    return (np.sum(data_rate) ** 2) / (n * np.sum(data_rate ** 2))


def validate(scenarios: list, solver_kwargs: dict) -> list:
    fairness_index = []

    for i, scenario in tqdm(enumerate(scenarios), total=len(scenarios)):
        configurations, _ = run_solver(scenario, solver_kwargs)

        data_rate = defaultdict(float)

        for conf, weight in configurations['shares'].items():
            for link, rate in configurations['link_rates'][conf].items():
                data_rate[link] += rate * weight

        data_rate = np.asarray(list(data_rate.values()))
        fairness_index.append(jains_fairness_index(data_rate))

    return fairness_index


if __name__ == '__main__':
    set_style()
    colors = get_cmap(3)

    distances = np.logspace(np.log10(4), 2, 100)
    scenarios = [simple_scenario_5(d) for d in distances]

    fairness_index_sum = validate(scenarios, {'opt_sum': True})
    fairness_index_min = validate(scenarios, {'opt_sum': False})

    with open('fairness.json', 'w') as f:
        json.dump({
            'distances': distances.tolist(),
            'fairness_index_sum': fairness_index_sum,
            'fairness_index_min': fairness_index_min
        }, f)

    plt.plot(distances, fairness_index_min, c=colors[1], linestyle='--', label='Solver w/ optimize min')
    plt.plot(distances, fairness_index_sum, c=colors[2], linestyle='--', label='Solver w/ optimize sum')
    plt.xscale('log')
    plt.xlabel(r'$d$ [m]')
    plt.xlim((distances.min(), distances.max()))
    plt.ylabel("Jain's fairness index")
    plt.ylim((-0.05, 1.05))
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fairness.pdf', bbox_inches='tight')
    plt.clf()
