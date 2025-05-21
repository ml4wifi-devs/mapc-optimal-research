from dataclasses import dataclass
from itertools import chain
from typing import Callable

import cloudpickle
import jax
import lz4.frame
import pulp as plp
from chex import Array
from mapc_sim.sim import network_data_rate
from mapc_sim.constants import DEFAULT_TX_POWER
from mapc_optimal import OptimizationType, Solver, positions_to_path_loss
from mapc_optimal.constants import MAX_TX_POWER, DATA_RATES
from tqdm import tqdm

from mapc_research.envs.scenario_impl import *
from mapc_research.mab.random_agent import RandomMapcAgent


SCENARIOS = [  # Note! The values are drawn from the interval [a, b) - a is inclusive, b is exclusive!
    (
        toy_scenario_1,
        {'d': (10, 51)}
    ),
    (
        toy_scenario_2,
        {'d_ap': (10, 41), 'd_sta': (1, 11)}
    ),
    (
        small_office_scenario,
        {'d_ap': (10, 41), 'd_sta': (1, 11)}
    ),
    (
        random_scenario,
        {'d_ap': (20, 101), 'n_ap': (2, 11), 'd_sta': (1, 9), 'n_sta_per_ap': (1, 6), 'randomize': (0, 1)}
    ),
    (
        residential_scenario,
        {'x_apartments': (2, 6), 'y_apartments': (2, 3), 'n_sta_per_ap': (1, 5), 'size': (5, 21)}
    ),
    (
        distance_scenario,
        {'d': (1, 51)}
    ),
    (
        hidden_station_scenario,
        {'d': (21, 51)}
    ),
    (
        flow_in_the_middle_scenario,
        {'d': (1, 31)}
    ),
    (
        dense_point_scenario,
        {'n_ap': (2, 11), 'n_associations': (1, 6)}
    ),
    (
        spatial_reuse_scenario,
        {'d_ap': (10, 21), 'd_sta': (1, 11)}
    ),
    (
        test_scenario,
        {'scale': (10, 31)}
    ),
    (
        indoor_small_bsss_scenario,
        {'grid_layers': (3, 4), 'n_sta_per_ap': (3, 11), 'frequency_reuse': (3, 4), 'bss_radius': (5, 21)}
    ),
]

N_TX_POWER_LEVELS = 4
TX_POWER_DELTA = 3.0
TX_POWER_LEVELS = jnp.array([MAX_TX_POWER - i * TX_POWER_DELTA for i in range(N_TX_POWER_LEVELS - 1, -1, -1)])


@dataclass
class TxPair:
    ap: int
    sta: int
    mcs: int
    tx_power: int


@dataclass
class Configuration:
    links: list[TxPair]


@dataclass
class DatasetItem:
    associations: dict
    pos: Array
    walls: Array
    path_loss_fn: Callable
    configurations: list[Configuration]


def save_dataset(dataset, path):
    with lz4.frame.open(path, 'wb') as f:
        f.write(cloudpickle.dumps(dataset))


def load_dataset(path):
    with lz4.frame.open(path, 'rb') as f:
        return cloudpickle.loads(f.read())


def draw_realizations(key, scenario_fn, param_ranges):
    *param_keys, seed_key = jax.random.split(key, len(param_ranges) + 1)
    params = {p: jax.random.randint(k, (), *v) for (p, v), k in zip(param_ranges.items(), param_keys)}
    seed = jax.random.randint(seed_key, (), 0, 2**30)
    (scenario, _), *_ = scenario_fn(seed=seed, **params).split_scenario()

    yield DatasetItem(
        associations=scenario.associations,
        pos=scenario.pos,
        walls=scenario.walls,
        path_loss_fn=scenario.path_loss_fn,
        configurations=[]
    )


def draw_scenarios(n_realizations, key, scenarios):
    n_params = list(map(len, [p for _, p in scenarios]))
    n_params = jnp.asarray(n_params)
    probs = n_params / n_params.sum()

    key, subkey = jax.random.split(key)
    scenario_idx = jax.random.choice(subkey, len(scenarios), p=probs, shape=(n_realizations,)).tolist()
    selected_scenarios = [scenarios[i] for i in scenario_idx]

    for i, (scenario, param_ranges) in enumerate(tqdm(selected_scenarios, desc='Scenarios')):
        key, subkey = jax.random.split(key)
        yield from draw_realizations(subkey, scenario, param_ranges)


def rate_to_mcs(mcs_data_rates, rate):
    return (jnp.abs(mcs_data_rates - rate)).argmin().item()


def tx_power_to_lvl(tx_power):
    return (jnp.abs(TX_POWER_LEVELS - tx_power)).argmin().item()


def peek_configuration(subkey, tx_matrix, pos, walls):
    ap, sta = jnp.where(tx_matrix)
    _, internals = jax.jit(network_data_rate)(subkey, tx_matrix, pos, None, jnp.full(tx_matrix.shape[0], DEFAULT_TX_POWER), 0.0, walls)
    return Configuration([TxPair(ap.item(), sta.item(), internals.mcs[ap].item(), 0) for i, (ap, sta) in enumerate(zip(ap, sta))])


def draw_configuration(n_configurations, key, dataset_item):
    associations = dataset_item.associations
    pos = dataset_item.pos
    walls = dataset_item.walls
    solver = RandomMapcAgent(associations)
    n = 0

    while n < 50:
        key, subkey = jax.random.split(key)
        tx_matrix, _ = solver.sample(None)

        if tx_matrix.sum() > 0:
            n += 1
            yield peek_configuration(subkey, tx_matrix, pos, walls)


def draw_history(n_configurations, key, dataset):
    for dataset_item in tqdm(dataset, desc='Configurations'):
        key, subkey = jax.random.split(key)
        dataset_item.configurations = list(draw_configuration(n_configurations, subkey, dataset_item))

    return dataset


def generate_dataset(seed, n_realizations, n_configurations, save_path):
    key = jax.random.PRNGKey(seed)
    scenarios_key, configurations_key = jax.random.split(key)

    dataset = list(draw_scenarios(n_realizations, scenarios_key, SCENARIOS))
    dataset = draw_history(n_configurations, configurations_key, dataset)

    save_dataset(dataset, save_path)


if __name__ == '__main__':
    generate_dataset(seed=42, n_realizations=1000, n_configurations=50, save_path='random_dataset.pkl.lz4')
    generate_dataset(seed=46, n_realizations=200, n_configurations=50, save_path='random_val_dataset.pkl.lz4')
