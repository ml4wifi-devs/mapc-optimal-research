from dataclasses import dataclass
from itertools import chain
from typing import Callable

import cloudpickle
import jax
import lz4.frame
import pulp as plp
from chex import Array
from mapc_optimal import Solver, positions_to_path_loss
from mapc_optimal.constants import DATA_RATES, MAX_TX_POWER

from mapc_research.envs.scenario_impl import *


SCENARIOS = [ # TODO: takes too long to generate
    # (
    #     residential_scenario,
    #     {'x_apartments': (2, 11), 'y_apartments': (2, 3), 'n_sta_per_ap': (1, 11), 'size': (5, 11)}
    # ),
    # (
    #     enterprise_scenario,
    #     {'x_offices': (1, 5), 'y_offices': (1, 3), 'x_cubicles': (8, 9), 'y_cubicles': (8, 9),
    #      'n_sta_per_cubicle': (1, 5), 'n_cubicle_per_ap': (16, 17), 'n_ap_per_office': (4, 5),
    #      'size_office': (20, 21), 'size_cubicle': (2, 3)}
    # ),
    # (
    #     indoor_small_bsss_scenario,
    #     {'grid_layers': (3, 4), 'n_sta_per_ap': (5, 31), 'frequency_reuse': (2, 4), 'bss_radius': (10, 11)}
    # ),
    (
        toy_scenario_1,
        {'d': (20, 51)}
    ),
    (
        toy_scenario_2,
        {'d_ap': (10, 41)}
    )
]

N_TX_POWER_LEVELS = 4
TX_POWER_DELTA = 3.0
TX_POWER_LEVELS = jnp.array([MAX_TX_POWER - i * TX_POWER_DELTA for i in range(N_TX_POWER_LEVELS - 1, -1, -1)])
DATA_RATES = jnp.array(DATA_RATES)


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


def draw_realizations(n_realizations, key, scenario, param_ranges):
    for _ in range(n_realizations):
        key, *subkey, seed_key = jax.random.split(key, len(param_ranges) + 2)
        params = {p: jax.random.randint(k, (), *v) for (p, v), k in zip(param_ranges.items(), subkey)}
        seed = jax.random.randint(seed_key, (), 0, 2**30)

        for subscenario, _ in scenario(seed=seed, **params).split_scenario():
            yield DatasetItem(
                associations=subscenario.associations,
                pos=subscenario.pos,
                walls=subscenario.walls,
                path_loss_fn=subscenario.path_loss_fn,
                configurations=[]
            )


def draw_scenarios(n_realizations, key, scenarios):
    for scenario, param_ranges in scenarios:
        key, subkey = jax.random.split(key)
        yield from draw_realizations(n_realizations, subkey, scenario, param_ranges)


def rate_to_mcs(rate):
    return (jnp.abs(DATA_RATES - rate)).argmin().item()


def tx_power_to_lvl(tx_power):
    return (jnp.abs(TX_POWER_LEVELS - tx_power)).argmin().item()


def peek_configuration(configurations, conf_idx):
    tx_pairs = [tx for tx in configurations['links'][conf_idx]]
    ap = [int(ap.split('_')[1]) for ap, sta in tx_pairs]
    sta = [int(sta.split('_')[1]) for ap, sta in tx_pairs]
    mcs = [rate_to_mcs(configurations['link_rates'][conf_idx][tx]) for tx in tx_pairs]
    tx_power = [tx_power_to_lvl(configurations['tx_power'][conf_idx][tx]) for tx in tx_pairs]
    return Configuration([TxPair(ap, sta, mcs[i], tx_power[i]) for i, (ap, sta) in enumerate(zip(ap, sta))])


def draw_configuration(n_configurations, key, dataset_item):
    associations = dataset_item.associations
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))
    path_loss = positions_to_path_loss(dataset_item.pos, dataset_item.walls)

    solver = Solver(stations, access_points, opt_sum=False, solver=plp.CPLEX_CMD(msg=False))
    configurations, _ = solver(path_loss, associations)

    shares = jnp.array(list(configurations['shares'].values()))
    idx_to_conf = {i: conf for i, conf in enumerate(configurations['links'])}

    for _ in range(n_configurations):
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, len(shares), p=shares).item()
        yield peek_configuration(configurations, idx_to_conf[idx])


def draw_history(n_configurations, key, dataset):
    from tqdm import tqdm
    for dataset_item in tqdm(dataset):
        key, subkey = jax.random.split(key)
        dataset_item.configurations = list(draw_configuration(n_configurations, subkey, dataset_item))

    return dataset


if __name__ == '__main__':
    seed = 42
    n_realizations = 10
    n_configurations = 10

    key = jax.random.PRNGKey(seed)
    scenarios_key, configurations_key = jax.random.split(key)

    dataset = list(draw_scenarios(n_realizations, scenarios_key, SCENARIOS))
    dataset = draw_history(n_configurations, configurations_key, dataset)
    save_dataset(dataset, 'dataset.pkl.lz4')
