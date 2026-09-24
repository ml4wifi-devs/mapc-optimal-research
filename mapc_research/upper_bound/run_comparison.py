import os
os.environ['JAX_ENABLE_X64'] = 'True'

import json
import signal
import subprocess
import time
from argparse import ArgumentParser
from itertools import chain

import jax
import numpy as np
import pulp as plp
from mapc_optimal import OptimizationType, PricingType, Solver, positions_to_path_loss
from mapc_optimal.utils import db_to_lin, lin_to_dbm
from tqdm import tqdm

from mapc_research.envs.scenario_impl import clustered_scenario, residential_scenario


GRIDS = [(2, 1), (3, 1), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5), (5, 6), (6, 6)]

EXPERIMENTS = [
    *[
        ('residential', x * y, 0, residential_scenario(
            seed=0, x_apartments=x, y_apartments=y, n_sta_per_ap=4, size=10., channel_width=80
        ))
        for x, y in GRIDS
    ],
    *[
        ('clustered', x * y, seed, clustered_scenario(
            seed=seed, n_ap=x * y, max_cluster=5, d_ap=7., d_cluster=75., d_sta=2., n_sta_per_ap=2, channel_width=80
        ))
        for x, y in GRIDS for seed in range(5)
    ]
]

CRITERIA = {
    't_optimal': OptimizationType.SUM,
    'f_optimal': OptimizationType.MAX_MIN,
    'l_optimal': OptimizationType.LEXICOGRAPHIC
}

METHODS = ['milp', 'tabu_model', 'tabu_sim']


def timeout_handler(signum, frame):
    raise TimeoutError


def make_evaluator(scenario, seed: int):
    n_nodes = scenario.pos.shape[0]
    default_tx_power = np.asarray(scenario.tx_power)

    def rates_fn(key, tx, tx_power, mcs):
        *_, internals = scenario(key, tx, tx_power, mcs, return_internals=True)
        return internals.average_data_rate / 1e6

    batched_rates_fn = jax.jit(jax.vmap(rates_fn, in_axes=(None, 0, 0, 0)))

    def evaluator(confs: list) -> list:
        start = time.time()
        tx = np.zeros((len(confs), n_nodes, n_nodes), dtype=int)
        tx_power = np.zeros((len(confs), n_nodes))
        mcs = np.zeros((len(confs), n_nodes), dtype=int)

        for c, conf in enumerate(confs):
            for (ap, sta), (power, m) in conf.items():
                a, s = int(ap.split('_')[1]), int(sta.split('_')[1])
                tx[c, a, s] = 1
                tx_power[c, a] = (default_tx_power[a] - lin_to_dbm(power)) / scenario.tx_power_delta
                mcs[c, a] = m

        evaluator.key, key = jax.random.split(evaluator.key)
        rates = np.asarray(batched_rates_fn(key, tx, tx_power, mcs))
        evaluator.time += time.time() - start
        evaluator.calls += len(confs)

        return [{l: float(rates[c, int(l[0].split('_')[1])]) for l in conf} for c, conf in enumerate(confs)]

    evaluator.key, evaluator.time, evaluator.calls = jax.random.PRNGKey(seed), 0., 0
    return evaluator


def model_rates(solver: Solver, result: dict, stations: list, link_path_loss: dict) -> dict:
    rates = {f'STA_{s}': 0. for s in stations}

    for c, share in result['shares'].items():
        tx_power = {l: db_to_lin(result['tx_power'][c][l]).item() for l in result['links'][c]}

        for l in tx_power:
            mcs = min(result['mcs'][c][l], solver.pricing._conf_mcs(l, tx_power, link_path_loss))
            rates[l[1]] += share * (float(solver.pricing.mcs_data_rates[mcs]) if mcs >= 0 else 0.)

    return rates


def simulator_rates(evaluator, result: dict, stations: list, n_draws: int = 100) -> dict:
    rates = {f'STA_{s}': 0. for s in stations}

    for c, share in result['shares'].items():
        conf = {l: (db_to_lin(result['tx_power'][c][l]).item(), result['mcs'][c][l]) for l in result['links'][c]}

        for _ in range(n_draws):
            for l, rate in evaluator([conf])[0].items():
                rates[l[1]] += share * rate / n_draws

    return rates


def run(scenario, opt_type: OptimizationType, method: str, seed: int, timeout: int) -> dict:
    associations = scenario.associations
    stations = list(chain.from_iterable(associations.values()))
    path_loss = positions_to_path_loss(scenario.pos, scenario.walls)
    evaluator = make_evaluator(scenario, seed)

    solver = Solver(
        stations, list(associations), 
        channel_width=scenario.channel_width, opt_type=opt_type, max_iterations=500,
        pricing_type=PricingType.MILP if method == 'milp' else PricingType.TABU,
        pricing_kwargs={'seed': seed, 'evaluator': evaluator if method == 'tabu_sim' else None},
        solver=plp.CPLEX_CMD(msg=False, options=['set mip tolerances integrality 0'], threads=16)
    )

    signal.alarm(timeout)
    start = time.time()

    try:
        result, _, objectives = solver(path_loss, associations, return_objectives=True)
    finally:
        signal.alarm(0)

    total_time = time.time() - start
    sim_time, sim_calls = evaluator.time, evaluator.calls
    link_path_loss = solver._generate_data(db_to_lin(path_loss), associations)['link_path_loss']
    model = model_rates(solver, result, stations, link_path_loss)
    simulator = simulator_rates(evaluator, result, stations)

    return {
        'time': total_time,
        'model_time': total_time - sim_time,
        'sim_calls': sim_calls,
        'iterations': len(objectives),
        'objectives': [float(o) for o in objectives],
        'converged': bool(objectives[-1] <= solver.epsilon),
        'model_rate': sum(model.values()),
        'model_station_rates': model,
        'sim_rate': sum(simulator.values()),
        'sim_station_rates': simulator,
        'configurations': [
            {
                'share': share,
                'links': [
                    {'ap': l[0], 'station': l[1], 'tx_power': result['tx_power'][c][l],
                     'mcs': result['mcs'][c][l], 'rate': result['link_rates'][c][l]}
                    for l in result['links'][c]
                ]
            }
            for c, share in result['shares'].items()
        ]
    }


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-t', '--timeout', type=int, default=3600)
    args.add_argument('-o', '--output', type=str, default='comparison_results.json')
    args = args.parse_args()

    signal.signal(signal.SIGALRM, timeout_handler)
    all_results, too_long = [], set()

    for family, n_ap, seed, scenario in tqdm(EXPERIMENTS, desc='Scenarios'):
        for criterion, opt_type in CRITERIA.items():
            for method in METHODS:
                if (family, criterion, method) in too_long:
                    continue

                try:
                    result = run(scenario, opt_type, method, seed, args.timeout)
                except TimeoutError:
                    subprocess.run(['pkill', '-P', str(os.getpid())])
                    result = {'timeout': True}
                    too_long.add((family, criterion, method))
                except Exception as e:
                    result = {'error': str(e)}

                all_results.append({
                    'family': family, 'n_ap': n_ap, 'seed': seed, 'criterion': criterion,
                    'method': method, **result
                })

                with open(args.output, 'w') as file:
                    json.dump(all_results, file, indent=4)
