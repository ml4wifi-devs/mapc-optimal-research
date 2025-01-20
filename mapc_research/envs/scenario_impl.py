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

    return StaticScenario(pos, mcs, associations, n_steps, str_repr="toy_scenario_1")


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

    return StaticScenario(pos, mcs, associations, n_steps, str_repr="toy_scenario_2")


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

    str_repr = f"small_office_{d_ap}_{d_sta}"

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

    return StaticScenario(pos, mcs, associations, n_steps, walls=walls, walls_pos=walls_pos, str_repr=str_repr)


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

    return OpenWifiScenario(pos, 4, associations, 500, str_repr="openwifi")


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

    str_repr = f"random_{seed}_{d_ap}_{n_ap}_{d_sta}_{n_sta_per_ap}"

    associations = {i: list(range(n_ap + i * n_sta_per_ap, n_ap + (i + 1) * n_sta_per_ap)) for i in range(n_ap)}

    key_first, key_sec = jax.random.split(jax.random.PRNGKey(seed), 2)
    pos_first = _draw_positions(key_first)
    pos_sec = _draw_positions(key_sec)

    return DynamicScenario(pos_first, mcs, associations, n_steps, pos_sec=pos_sec, switch_steps=[n_steps // 2], str_repr=str_repr)


def residential_scenario(
        seed: int,
        n_steps: int,
        x_apartments: int,
        y_apartments: int,
        n_sta_per_ap: int,
        size: Scalar,
        mcs: int = 11
) -> StaticScenario:
    """
    Implementation of the Residential Scenario from S. Merlin et al. "TGax Simulation Scenarios", IEEE 802.11-14/0980r16

    The path loss model of this scenario requires:

    BREAKING_POINT = 5
    WALL_LOSS = 5

    """

    key = jax.random.PRNGKey(seed)
    str_repr = f"residential_{seed}_{x_apartments}_{y_apartments}_{n_sta_per_ap}_{size}"
    associations, pos, walls_pos = {}, [], []
    rooms = {}

    for x, y in product(range(x_apartments), range(y_apartments)):
        ap, stas = len(pos), list(range(len(pos) + 1, len(pos) + n_sta_per_ap + 1))
        associations[ap] = stas
        rooms.update({node: (x, y) for node in stas + [ap]})

        walls_pos.append([x * size, y * size, (x + 1) * size, y * size])
        walls_pos.append([x * size, y * size, x * size, (y + 1) * size])

        pos_key, key = jax.random.split(key)
        pos += (jax.random.uniform(pos_key, (n_sta_per_ap + 1, 2)) * size + jnp.array([x * size, y * size])).tolist()

    walls_pos.append([x_apartments * size, 0, x_apartments * size, y_apartments * size])
    walls_pos.append([0, y_apartments * size, x_apartments * size, y_apartments * size])
    walls = jnp.zeros((len(pos), len(pos)))

    for i, j in product(rooms.keys(), repeat=2):
        xi, yi = rooms[i]
        xj, yj = rooms[j]

        walls = walls.at[i, j].set(jnp.abs(xi - xj) + jnp.abs(yi - yj))
        walls = walls.at[j, i].set(jnp.abs(xi - xj) + jnp.abs(yi - yj))

    return StaticScenario(jnp.array(pos), mcs, associations, n_steps, walls=walls, walls_pos=jnp.array(walls_pos), str_repr=str_repr)


def distance_scenario(
        d: Scalar,
        n_steps: int,
        mcs: int = 11
) -> StaticScenario:
    """
    There is a single AP with a single STA placed at distance `d`. 
    """
    
    return StaticScenario(jnp.array([[0., 0.], [d, 0.]]), mcs, {0: [1]}, n_steps, str_repr=f"distance_{d}")


def hidden_station_scenario(
        d: Scalar,
        n_steps: int,
        mcs: int = 4
) -> StaticScenario:
    """
    There are two APs 2 distance units `d` apart. Both APs have a single
    station placed in between them in the same place.

    AP_A <--d--> STA_1, STA_2 <--d--> AP_B 
    """

    pos = jnp.array([
        [0., 0.],       # AP A
        [d, 0.],        # STA 1
        [d, 0.],        # STA 2
        [2 * d, 0.]     # AP B
    ])

    associations = {
        0: [1],
        3: [2]
    }

    return StaticScenario(pos, mcs, associations, n_steps, str_repr=f"hidden_station_{d}")


def flow_in_the_middle_scenario(
        d: Scalar,
        n_steps: int,
        mcs: int = 4,
) -> StaticScenario:
    """
    There are thres APs placed in line spaced `d` units apart. Each AP is associated with a single STA,
    placed in the same place as the AP.

    AP_A <--d--> STA_1, STA_2 <--d--> AP_B 
    """

    pos = jnp.array([
        [0., 0.],       # AP A
        [0., 0.],       # STA 1
        [d, 0.],        # AP B
        [d, 0.],        # STA 2
        [2 * d, 0.],    # AP C
        [2 * d, 0.]     # STA 3
    ])

    associations = {
        0: [1],
        2: [3],
        4: [5]
    }

    return StaticScenario(pos, mcs, associations, n_steps, str_repr=f"flow_in_the_middle_{d}")


def dense_point_scenario(
        n_ap: int,
        n_associations: int,
        n_steps: int,
        mcs: int = 11,
) -> StaticScenario:
    """
    There is `n_ap` APs with `n_associations` STAs each. All of the devices are placed at the same point. 
    """

    pos = jnp.array([[0., 0.] for _ in range(n_ap * (n_associations + 1))])

    associations = {i: [n_ap + i * n_associations + j for j in range(n_associations)] for i in range(n_ap)}
    
    return StaticScenario(pos, mcs, associations, n_steps, str_repr=f"dense_point_{n_ap}_{n_associations}")


def spatial_reuse_scenario(d_ap: Scalar, d_sta: Scalar, mcs: int = 7, n_steps: int = 600) -> StaticScenario:
    """
    STA 1 <--d_sta--> AP A <--d_ap--> AP B <--d_sta--> STA 4
    """

    pos = jnp.array([
        [0., 0.],               # STA 1
        [d_sta, 0.],            # AP A
        [d_sta + d_ap, 0.],     # AP B
        [2*d_sta + d_ap, 0.]    # STA 2
    ])

    associations = {
        1: [0],
        2: [3]
    }

    return StaticScenario(pos, mcs, associations, n_steps, str_repr="spatial_reuse_scenario")


def test_scenario(scale: float = 1.0) -> StaticScenario:
    """

             STA 1     AP A     STA 2


    -----------------------------------------



    AP B    STA 3               STA 4    AP C

    """

    pos = scale * jnp.array([
        [ 0.,  1.],  # AP A
        [-1., -1.],  # AP B
        [ 1., -1.],  # AP C
        [-0.5,  1.],  # STA 1
        [ 0.5,  1.],  # STA 2
        [-0.5, -1.],  # STA 3
        [ 0.5, -1.],  # STA 4
    ])

    associations = {
        0: [3, 4],
        1: [5],
        2: [6]
    }

    walls_pos = scale * jnp.array([
        [-2.0, 0.0, 2.0, 0.0]
    ])

    return StaticScenario(pos, 0, associations, 0, walls_pos=walls_pos)

def enterprise_scenario(
        seed: int,
        n_steps: int,
        x_offices: int = 1, #TODO update to 4
        y_offices: int = 1, #TODO update to 2
        x_cubicles: int = 8,
        y_cubicles: int = 8,
        n_sta_per_cubicle: int = 1, #TODO update to 4
        n_cubicle_per_ap: int = 4, #TODO update to 16
        n_ap_per_office: int = 4,
        size_office: Scalar = 20,
        size_cubicle: Scalar = 2,
        mcs: int = 11
) -> StaticScenario:
    """
    Implementation of the Enterprise Scenario from S. Merlin et al. "TGax Simulation Scenarios", IEEE 802.11-14/0980r16

    The path loss model of this scenario requires:

    BREAKING_POINT = 10
    WALL_LOSS = 7

    """

    # Additional variables describing the scenario
    inner_corridor_width = (size_office - x_cubicles * size_cubicle) / 4
    outer_corridor_width = inner_corridor_width/2

    key = jax.random.PRNGKey(seed)
    str_repr = f"enterprise_{seed}_{x_offices}_{y_offices}_{n_cubicle_per_ap}_{n_sta_per_cubicle}_{size_office}_{size_cubicle}"
    associations, pos, walls_pos = {}, [], []
    offices, cubicles = {}, {}

    office_counter = 0
    for x, y in product(range(x_offices), range(y_offices)):
        aps = list(range(office_counter*n_ap_per_office, office_counter*n_ap_per_office+n_ap_per_office))
        for ap in aps:
            associations[ap]=list(range(len(aps)+ap*n_cubicle_per_ap*n_sta_per_cubicle, len(aps)+ap*n_cubicle_per_ap*n_sta_per_cubicle+n_cubicle_per_ap*n_sta_per_cubicle))

        ap_pos = (jnp.array([[ 5,  5],
                            [15,  5],
                            [ 5, 15],
                            [15, 15]])
                  + jnp.array([x * size_office, y * size_office]))

        walls_pos.append([x * size_office, y * size_office, (x + 1) * size_office, y * size_office])
        walls_pos.append([x * size_office, y * size_office, x * size_office, (y + 1) * size_office])

        pos += ap_pos.tolist()
        for cubicle in range(n_cubicle_per_ap):
            pos_key, key = jax.random.split(key)
            pos += (jax.random.uniform(pos_key, (n_cubicle_per_ap*n_sta_per_cubicle, 2)) * size_office + jnp.array([x * size_office, y * size_office])).tolist()

        office_counter+=1

    print(associations)

    print(len(pos))

    walls_pos.append([x_offices * size_office, 0, x_offices * size_office, y_offices * size_office])
    walls_pos.append([0, y_offices * size_office, x_offices * size_office, y_offices * size_office])


    return StaticScenario(jnp.array(pos), mcs, associations, n_steps, walls_pos=jnp.array(walls_pos), str_repr=str_repr)