from itertools import product

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, Scalar

from mapc_research.envs.static_scenario import StaticScenario
from mapc_research.envs.dynamic_scenario import DynamicScenario


def toy_scenario_1(d: Scalar = 20., mcs: int = 7, n_steps: int = 600) -> StaticScenario:
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

    return StaticScenario(pos, mcs, associations, n_steps)


def toy_scenario_2(d_ap: Scalar = 50., d_sta: Scalar = 2., mcs: int = 11, n_steps : int = 600) -> StaticScenario:
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

    return StaticScenario(pos, mcs, associations, n_steps)


def small_office_scenario(d_ap: Scalar, d_sta: Scalar, n_steps, mcs: int = 11) -> StaticScenario:
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

    return StaticScenario(pos, mcs, associations, n_steps, walls=walls, walls_pos=walls_pos)


def openwifi_scenario():
    class OpenWifiScenario(StaticScenario):
        def __call__(self, key, tx, tx_power):
            _, reward = super().__call__(key, tx, tx_power)
            return int(10 * reward), reward

    pos = jnp.asarray([
        [-47., 0.],  # AP1
        [0., 39.],   # AP2
        [34., 0.],   # AP3
        [-29., 0.],  # STA1
        [-1., 0.],   # STA2
        [0., 25.],   # STA3
        [0., 1.],    # STA4
        [19., 0.],   # STA5
        [1., 0.],    # STA6
    ])

    associations = {
        0: [3, 4],
        1: [5, 6],
        2: [7, 8],
    }

    return OpenWifiScenario(pos, 4, associations, 500)


def random_scenario(
        seed: int,
        n_steps: int,
        d_ap: float,
        n_ap: int,
        d_sta: float,
        n_sta_per_ap: int,
        mcs: int = 11
) -> DynamicScenario:
    def _draw_positions(key: PRNGKey) -> Array:
        ap_key, key = jax.random.split(key)
        ap_pos = jax.random.uniform(ap_key, (n_ap, 2)) * d_ap
        sta_pos = []

        for pos in ap_pos:
            sta_key, key = jax.random.split(key)
            center = jnp.repeat(pos[None, :], n_sta_per_ap, axis=0)
            stations = center + jax.random.normal(sta_key, (n_sta_per_ap, 2)) * d_sta
            sta_pos += stations.tolist()

        pos = jnp.array(ap_pos.tolist() + sta_pos)
        return pos

    associations = {i: list(range(n_ap + i * n_sta_per_ap, n_ap + (i + 1) * n_sta_per_ap)) for i in range(n_ap)}

    key_first, key_sec = jax.random.split(jax.random.PRNGKey(seed), 2)
    pos_first = _draw_positions(key_first)
    pos_sec = _draw_positions(key_sec)

    return DynamicScenario(pos_first, mcs, associations, n_steps, pos_sec=pos_sec, switch_steps=[n_steps // 2])


def residential_scenario(
        seed: int,
        n_steps: int,
        x_apartments: int,
        y_apartments: int,
        n_sta_per_ap: int,
        size: Scalar,
        mcs: int = 11
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

    return StaticScenario(jnp.array(pos), mcs, associations, n_steps, walls_pos=jnp.array(walls_pos))
