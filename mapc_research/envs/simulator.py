import os
os.environ['JAX_ENABLE_X64'] = "True"

import time
from itertools import chain

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mapc_optimal import Solver, positions_to_path_loss
from mapc_sim.constants import DATA_RATES, DEFAULT_SIGMA
from mapc_sim.sim import network_data_rate
from tqdm import tqdm

from mapc_research.envs.static_scenarios import *
from mapc_research.plots import set_style
from mapc_research.plots.utils import confidence_interval


RATE_TO_MCS = dict(zip(DATA_RATES.tolist(), range(len(DATA_RATES))))
RATE_TO_MCS[0.] = 0


def run_solver(scenario: StaticScenario, obj_filename: str = None, solver_kwargs: dict = None) -> tuple:
    associations = scenario.get_associations()
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))

    positions = scenario.pos
    walls = scenario.walls
    path_loss = positions_to_path_loss(positions, walls)

    solver = Solver(stations, access_points, **(solver_kwargs or {}))
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


def run_simulation(scenario: StaticScenario, configurations: dict, iters: int = 5000) -> list:
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

        for link in configurations['link_rates'][conf]:
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

    return results


if __name__ == '__main__':
    solver_kwargs = {
        'opt_sum': True
    }

    distances = jnp.logspace(jnp.log10(4), 2, 30)
    scenario_type = simple_scenario_5
    scenario_name = scenario_type.__name__

    solver_results = []
    simulator_mean = []
    simulator_std_low = []
    simulator_std_high = []

    start = time.time()

    for d in tqdm(distances):
        scenario = scenario_type(d)

        configurations, solver_rate = run_solver(scenario, f'{scenario_name}_obj_{d:.2f}.pdf', solver_kwargs)
        simulator_rate = run_simulation(scenario, configurations)

        mean, ci_low, ci_high = confidence_interval(jnp.asarray(simulator_rate))
        simulator_mean.append(mean)
        simulator_std_low.append(ci_low)
        simulator_std_high.append(ci_high)
        solver_results.append(solver_rate)

    print(f'Total time: {time.time() - start:.2f} s')

    plt.plot(distances, solver_results, c='C0', label='Solver')
    plt.plot(distances, simulator_mean, c='C1', label='Simulator')
    if os.path.exists('alignment-distances.npy'):
        plt.plot(
            jnp.load('alignment-distances.npy'), 
            jnp.load('alignment-upper-bound.npy'), c='tab:red', label="Alignment"
        )
        plt.plot(
            jnp.load('alignment-distances.npy'), 
            jnp.load('alignment-mean-single.npy'), c='tab:gray', label="One AP", linestyle='--'
        )
    plt.fill_between(distances, simulator_std_low, simulator_std_high, alpha=0.3, color='C1', linewidth=0)
    plt.xscale('log')
    plt.xlabel(r'$d$ [m]')
    plt.ylabel('Effective data rate [Mb/s]')
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{scenario_name}.pdf', bbox_inches='tight')
    plt.clf()
