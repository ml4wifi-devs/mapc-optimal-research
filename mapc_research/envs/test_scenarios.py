from functools import partial
from itertools import product
from typing import Optional

import jax
import jax.numpy as jnp
from chex import Scalar
from mapc_sim.constants import DEFAULT_TX_POWER, DEFAULT_SIGMA

from mapc_research.envs.scenario import StaticScenario


def toy_scenario_1(
        d: Scalar = 40.,
        mcs: int = 11,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    """
    STA 1     AP A     STA 2     STA 3     AP B     STA 4
    """

    pos = jnp.array([
        [0 * d, 0.],  # STA 1
        [1 * d, 0.],  # AP A
        [2 * d, 0.],  # STA 2
        [3 * d, 0.],  # STA 3
        [4 * d, 0.],  # AP B
        [5 * d, 0.]   # STA 4
    ])

    associations = {
        1: [0, 2],
        4: [3, 5]
    }

    return StaticScenario(pos, mcs, tx_power, sigma, associations)


def toy_scenario_2(
        d_ap: Scalar = 50.,
        d_sta: Scalar = 1.,
        mcs: int = 11,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    """
    STA 16   STA 15                  STA 12   STA 11

         AP D                             AP C

    STA 13   STA 14                  STA 9    STA 10



    STA 4    STA 3                   STA 8    STA 7

         AP A                             AP B

    STA 1    STA 2                   STA 5    STA 6
    """

    ap_pos = [
        [0 * d_ap, 0 * d_ap],  # AP A
        [1 * d_ap, 0 * d_ap],  # AP B
        [1 * d_ap, 1 * d_ap],  # AP C
        [0 * d_ap, 1 * d_ap],  # AP D
    ]

    dx = jnp.array([-1, 1, 1, -1]) * d_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_sta / jnp.sqrt(2)

    sta_pos = [[x + dx[i], y + dy[i]] for x, y in ap_pos for i in range(len(dx))]
    pos = jnp.array(ap_pos + sta_pos)

    associations = {
        0: [4, 5, 6, 7],
        1: [8, 9, 10, 11],
        2: [12, 13, 14, 15],
        3: [16, 17, 18, 19]
    }

    return StaticScenario(pos, mcs, tx_power, sigma, associations)


def small_office_scenario(
        d_ap: Scalar,
        d_sta: Scalar,
        mcs: int = 11,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    """
    STA 16   STA 15         |        STA 12   STA 11
                            |
         AP D               |              AP C
                            |
    STA 13   STA 14         |         STA 9    STA 10
                            |
    ------------------------+------------------------

    STA 4    STA 3                    STA 8    STA 7

         AP A                              AP B

    STA 1    STA 2                    STA 5    STA 6
    """

    ap_pos = [
        [0 * d_ap, 0 * d_ap],  # AP A
        [1 * d_ap, 0 * d_ap],  # AP B
        [1 * d_ap, 1 * d_ap],  # AP C
        [0 * d_ap, 1 * d_ap],  # AP D
    ]

    dx = jnp.array([-1, 1, 1, -1]) * d_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_sta / jnp.sqrt(2)

    sta_pos = [[x + dx[i], y + dy[i]] for x, y in ap_pos for i in range(len(dx))]
    pos = jnp.array(ap_pos + sta_pos)

    associations = {
        0: [4, 5, 6, 7],
        1: [8, 9, 10, 11],
        2: [12, 13, 14, 15],
        3: [16, 17, 18, 19]
    }

    aps = associations.keys()

    # Setup walls in between each BSS
    walls = jnp.zeros((20, 20))
    walls = walls.at[4:, 4:].set(True)
    for i in range(20):
        for j in range(20):

            # If both are APs
            if i in aps and j in aps:
                walls = walls.at[i, j].set(i != j)

            # If i is an AP
            elif i in aps:
                for ap_j in set(aps) - {i}:
                    for sta in associations[ap_j]:
                        walls = walls.at[i, sta].set(True)

            # If j is an AP
            elif j in aps:
                for ap_i in set(aps) - {j}:
                    for sta in associations[ap_i]:
                        walls = walls.at[sta, j].set(True)

            # If both are STAs
            else:
                for ap in aps:
                    if i in associations[ap] and j in associations[ap]:
                        walls = walls.at[i, j].set(False)

    # - Remove wall between AP A and AP B
    walls = walls.at[:2, :2].set(False)
    walls = walls.at[1, 4:8].set(False)
    walls = walls.at[4:8, 1].set(False)
    walls = walls.at[0, 8:12].set(False)
    walls = walls.at[8:12, 0].set(False)
    walls = walls.at[4:12, 4:12].set(False)

    # Walls positions
    walls_pos = jnp.array([
        [-d_ap / 2, d_ap / 2, d_ap + d_ap / 2, d_ap / 2],
        [d_ap / 2, d_ap / 2, d_ap / 2, d_ap + d_ap / 2],
    ])

    return StaticScenario(pos, mcs, tx_power, sigma, associations, walls, walls_pos)


small_office_scenario_10 = partial(small_office_scenario, d_ap=10., d_sta=2.)
small_office_scenario_20 = partial(small_office_scenario, d_ap=20., d_sta=2.)
small_office_scenario_30 = partial(small_office_scenario, d_ap=30., d_sta=2.)


def random_scenario(
        seed: int,
        d_ap: Optional[Scalar] = 100.,
        n_ap: int = 4,
        d_sta: Scalar = 1.,
        n_sta_per_ap: int = 4,
        ap_density: Optional[float] = None,
        mcs: int = 11,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    assert d_ap is not None or ap_density is not None, 'Either d_ap or ap_density must be specified'
    if ap_density is not None:
        d_ap = jnp.sqrt(n_ap / ap_density)  # As AP density is constant, d_ap is proportional to sqrt(n_ap)
    ap_key, key = jax.random.split(jax.random.PRNGKey(seed))
    ap_pos = jax.random.uniform(ap_key, (n_ap, 2)) * d_ap
    sta_pos = []

    for pos in ap_pos:
        sta_key, key = jax.random.split(key)
        center = jnp.repeat(pos[None, :], n_sta_per_ap, axis=0)
        stations = center + jax.random.normal(sta_key, (n_sta_per_ap, 2)) * d_sta
        sta_pos += stations.tolist()

    pos = jnp.array(ap_pos.tolist() + sta_pos)
    associations = {i: list(range(n_ap + i * n_sta_per_ap, n_ap + (i + 1) * n_sta_per_ap)) for i in range(n_ap)}

    return StaticScenario(pos, mcs, tx_power, sigma, associations)


def residential_scenario(
        seed: int,
        x_apartments: int = 4,
        y_apartments: int = 4,
        n_sta_per_ap: int = 4,
        size: Scalar = 10.,
        mcs: int = 11,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    key = jax.random.PRNGKey(seed)
    associations, pos, walls_pos = {}, [], []

    for i, j in product(range(x_apartments), range(y_apartments)):
        associations[len(pos)] = list(range(len(pos) + 1, len(pos) + n_sta_per_ap + 1))
        walls_pos.append([i * size, j * size, (i + 1) * size, j * size])
        walls_pos.append([i * size, j * size, i * size, (j + 1) * size])

        pos_key, key = jax.random.split(key)
        pos += (jax.random.uniform(pos_key, (n_sta_per_ap + 1, 2)) * size + jnp.array([i * size, j * size])).tolist()

    walls_pos.append([x_apartments * size, 0, x_apartments * size, y_apartments * size])
    walls_pos.append([0, y_apartments * size, x_apartments * size, y_apartments * size])

    return StaticScenario(jnp.array(pos), mcs, tx_power, sigma, associations, walls_pos=jnp.array(walls_pos))
