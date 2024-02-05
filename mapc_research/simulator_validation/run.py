import os
os.environ['JAX_ENABLE_X64'] = "True"

import json
from functools import partial
from itertools import chain
from typing import Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mapc_mab.agents import MapcAgentFactory
from mapc_mab.envs.run import run_scenario as run_mab_scenario
from mapc_optimal import Solver, positions_to_path_loss
from mapc_sim.constants import DATA_RATES, DEFAULT_SIGMA
from mapc_sim.sim import network_data_rate
from reinforced_lib.agents.mab import NormalThompsonSampling
from tqdm import tqdm

from mapc_research.envs.static_scenarios import simple_scenario_5, StaticScenario
from mapc_research.plots import confidence_interval, get_cmap, set_style


RATE_TO_MCS = dict(zip(DATA_RATES.tolist(), range(len(DATA_RATES))))
RATE_TO_MCS[0.] = 0


def run_solver(scenario: StaticScenario, obj_filename: Optional[str], solver_kwargs: dict) -> tuple:
    associations = scenario.get_associations()
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))

    positions = scenario.pos
    walls = scenario.walls
    path_loss = positions_to_path_loss(positions, walls)

    solver = Solver(stations, access_points, **solver_kwargs)
    configurations, rate, objectives = solver(path_loss, return_objectives=True)

    if obj_filename:
        plt.plot(objectives)
        plt.xlabel('Iteration')
        plt.ylabel('Pricing objective')
        plt.yscale('log')
        plt.grid()
        plt.tight_layout()
        plt.savefig(obj_filename, bbox_inches='tight')
        plt.clf()

    return configurations, rate


def run_simulation(scenario: StaticScenario, configurations: dict, iters: int) -> jnp.ndarray:
    associations = scenario.get_associations()
    n_nodes = len(associations.keys()) + len(list(chain.from_iterable(associations.values())))

    key = jax.random.PRNGKey(42)
    sim_fn = jax.jit(partial(
        network_data_rate,
        pos=scenario.pos,
        sigma=DEFAULT_SIGMA,
        walls=scenario.walls
    ))

    sim_conf = []
    sim_conf_prob = []

    for conf in configurations['shares']:
        tx = jnp.zeros((n_nodes, n_nodes), dtype=float)

        for link in configurations['links'][conf]:
            ap = int(link[0].split('_')[1])
            sta = int(link[1].split('_')[1])
            tx = tx.at[ap, sta].set(1.)

        mcs = jnp.zeros(n_nodes, dtype=int)

        for link, rate in configurations['link_rates'][conf].items():
            ap = int(link[0].split('_')[1])
            mcs = mcs.at[ap].set(RATE_TO_MCS[rate])

        tx_power = jnp.zeros(n_nodes, dtype=float)

        for link, power in configurations['tx_power'][conf].items():
            ap = int(link[0].split('_')[1])
            tx_power = tx_power.at[ap].set(power)

        sim_conf.append(dict(tx=tx, mcs=mcs, tx_power=tx_power))
        sim_conf_prob.append(configurations['shares'][conf])

    sim_conf_prob = jnp.cumsum(jnp.asarray(sim_conf_prob), dtype=float)
    results = []

    for _ in range(iters):
        sim_key, conf_key, key = jax.random.split(key, 3)
        conf_idx = (sim_conf_prob <= jax.random.uniform(conf_key)).sum()
        conf = sim_conf[conf_idx]
        results.append(sim_fn(key=sim_key, tx=conf['tx'], mcs=conf['mcs'], tx_power=conf['tx_power']))

    return jnp.asarray(results)


def validate(scenarios: list, obj_filename: Optional[str], solver_kwargs: dict) -> tuple:
    solver_results = []
    simulator_results = []

    for i, scenario in enumerate(tqdm(scenarios)):
        configurations, solver_rate = run_solver(scenario, f'{obj_filename}_{i}.pdf' if obj_filename else None, solver_kwargs)

        if configurations:
            simulator_rate = run_simulation(scenario, configurations, 5000)
        else:
            simulator_rate = [0.]

        simulator_results.append(confidence_interval(simulator_rate))
        solver_results.append(solver_rate)

    return jnp.asarray(solver_results), jnp.asarray(simulator_results)


def run_mab(scenarios: list, n_reps: int, n_steps: int, seed: int, n_drop: int) -> jnp.ndarray:
    results = []

    for scenario in tqdm(scenarios):
        associations = scenario.get_associations()
        agent_factory = MapcAgentFactory(associations, NormalThompsonSampling, {
            'alpha': 1.6849910402998838,
            'beta': 308.35636094889753,
            'lam': 0.0002314863115510709,
            'mu': 263.8448112411686
        })

        runs, _ = run_mab_scenario(agent_factory, scenario, n_reps, n_steps, seed)
        runs = jnp.asarray(runs)[:, n_drop:].reshape(-1)
        results.append(confidence_interval(runs))

    return jnp.asarray(results)


if __name__ == '__main__':
    set_style()
    colors = get_cmap(5)[1:]

    distances = jnp.logspace(jnp.log10(4), 2, 100)
    scenarios = [simple_scenario_5(d) for d in distances]

    opt_sum_solver, opt_sum_simulator = validate(scenarios, None, {'opt_sum': True})
    opt_min_solver, opt_min_simulator = validate(scenarios, None, {'opt_sum': False})
    mab = run_mab(scenarios, 5, 2000, 42, 1000)

    with open('simulator_validation.json', 'w') as f:
        json.dump({
            'distances': distances.tolist(),
            'opt_sum_solver': opt_sum_solver.tolist(),
            'opt_sum_simulator': opt_sum_simulator.tolist(),
            'opt_min_solver': opt_min_solver.tolist(),
            'opt_min_simulator': opt_min_simulator.tolist(),
            'mab': mab.tolist()
        }, f)

    plt.plot(distances, opt_sum_solver, c=colors[3], label='Solver [opt. sum]')
    plt.plot(distances, opt_sum_simulator[:, 0], c=colors[3], label='Simulator [opt. sum]', linestyle='--')
    plt.fill_between(distances, opt_sum_simulator[:, 1], opt_sum_simulator[:, 2], color=colors[3], alpha=0.3, linewidth=0)
    plt.plot(distances, mab[:, 0], c=colors[1], label='MAB w/ TS', linestyle='--')
    plt.fill_between(distances, mab[:, 1], mab[:, 2], color=colors[1], alpha=0.3, linewidth=0)
    plt.plot(distances, opt_min_solver, c=colors[2], label='Solver [opt. min]')
    plt.plot(distances, opt_min_simulator[:, 0], c=colors[2], label='Simulator [opt. min]', linestyle='--')
    plt.fill_between(distances, opt_min_simulator[:, 1], opt_min_simulator[:, 2], color=colors[2], alpha=0.3, linewidth=0)
    plt.xscale('log')
    plt.xlabel(r'$d$ [m]')
    plt.ylabel('Effective data rate [Mb/s]')
    plt.ylim((0, 800))
    plt.legend(ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig('simulator_validation.pdf', bbox_inches='tight')
    plt.clf()
