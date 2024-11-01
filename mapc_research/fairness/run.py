import os
os.environ['JAX_ENABLE_X64'] = "True"

import json
from collections import defaultdict
from functools import partial
from itertools import chain

import jax
import numpy as np
import matplotlib.pyplot as plt
from mapc_mab.agents import MapcAgentFactory
from mapc_optimal import Solver, positions_to_path_loss
from mapc_sim.constants import DEFAULT_SIGMA, FRAME_LEN, TAU
from mapc_sim.sim import network_data_rate
from reinforced_lib.agents.mab import NormalThompsonSampling
from tqdm import tqdm

from mapc_research.envs.static_scenario import StaticScenario
from mapc_research.envs.scenario_impl import small_office_scenario
from mapc_research.plots.config import get_cmap, set_style


FRAMES_TO_RATE = np.array(FRAME_LEN / (1e6 * TAU))


def jains_fairness_index(data_rate: np.ndarray) -> float:
    n = len(data_rate)
    return (np.sum(data_rate) ** 2) / (n * np.sum(data_rate ** 2))


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


def solver_fairness(scenarios: list, solver_kwargs: dict) -> list:
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


def run_mab(scenario: StaticScenario, n_reps: int, n_steps: int, seed: int) -> list:
    key = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    ts_params = {
        'alpha': 1.6849910402998838,
        'beta': 308.35636094889753,
        'lam': 0.0002314863115510709,
        'mu': 263.8448112411686
    }
    agent_factory = MapcAgentFactory(
        associations=scenario.get_associations(),
        agent_type=NormalThompsonSampling,
        agent_params_lvl1=ts_params,
        agent_params_lvl2=ts_params,
        agent_params_lvl3=ts_params,
        hierarchical=True
    )


    data_rate_fn = jax.jit(partial(
        network_data_rate,
        pos=scenario.pos,
        mcs=np.full(scenario.pos.shape[0], scenario.mcs, dtype=np.int32),
        tx_power=scenario.tx_power,
        sigma=DEFAULT_SIGMA,
        walls=scenario.walls,
        return_sample=True
    ))

    runs = []

    for i in range(n_reps):
        agent = agent_factory.create_mapc_agent()
        runs.append([])

        for j in range(n_steps):
            key, scenario_key = jax.random.split(key)
            tx, _ = agent.sample()

            reward, frames_transmitted = data_rate_fn(scenario_key, tx)
            data_rate = np.array(frames_transmitted) * FRAMES_TO_RATE
            agent.update([reward])

            for ap, sta in zip(*np.where(tx)):
                data_rate[sta] = data_rate[ap]
                data_rate[ap] = 0.

            runs[-1].append(data_rate)

    return runs


def mab_fairness(scenarios: list, n_reps: int, n_steps: int, seed: int, n_drop: int) -> list:
    fairness_index = []

    for i, scenario in tqdm(enumerate(scenarios), total=len(scenarios)):
        stations = np.asarray(list(chain.from_iterable(scenario.get_associations().values())))
        data_rate = run_mab(scenario, n_reps, n_steps, seed)
        data_rate = np.asarray(data_rate)[:, n_drop:, :].mean(axis=(0, 1))[stations]
        fairness_index.append(jains_fairness_index(data_rate))

    return fairness_index


if __name__ == '__main__':
    set_style()
    colors = get_cmap(3)

    distances = np.logspace(np.log10(4), 2, 100)
    scenarios = [small_office_scenario(d, 2.0) for d in distances]

    fairness_index_sum = solver_fairness(scenarios, {'opt_sum': True})
    fairness_index_min = solver_fairness(scenarios, {'opt_sum': False})
    fairness_index_mab = mab_fairness(scenarios, 5, 2000, 42, 1000)

    with open('fairness.json', 'w') as f:
        json.dump({
            'distances': distances.tolist(),
            'fairness_index_sum': fairness_index_sum,
            'fairness_index_min': fairness_index_min,
            'fairness_index_mab': fairness_index_mab
        }, f)

    plt.plot(distances, fairness_index_sum, c=colors[2], label='Solver w/ optimize sum')
    plt.plot(distances, fairness_index_min, c=colors[1], label='Solver w/ optimize min')
    plt.plot(distances, fairness_index_mab, c=colors[0], label='MAB w/ TS', linestyle='--')
    plt.xscale('log')
    plt.xlabel(r'$d$ [m]')
    plt.xlim((distances.min(), distances.max()))
    plt.ylabel("Jain's fairness index")
    plt.ylim((0, 1.05))
    plt.grid(axis='y')
    plt.legend(loc=(0.02, 0.28))
    plt.tight_layout()
    plt.savefig('fairness.pdf', bbox_inches='tight')
    plt.clf()
