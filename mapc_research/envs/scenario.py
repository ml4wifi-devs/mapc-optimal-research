import string
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from itertools import product
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from chex import Array, Scalar, PRNGKey
from mapc_sim.constants import DEFAULT_SIGMA, DEFAULT_TX_POWER
from mapc_sim.sim import network_data_rate
from mapc_sim.utils import tgax_path_loss as path_loss

from mapc_research.plots import get_cmap


class Scenario(ABC):
    """
    Base class for scenarios.

    Parameters
    ----------
    associations: Dict
        Dictionary of associations between access points and stations.
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    walls: Optional[Array]
        Matrix counting the walls between each pair of nodes.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to the X and Y coordinates of
        the wall start and end points.
    """

    def __init__(
            self,
            associations: Dict,
            pos: Array,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None
    ) -> None:
        self.CCA_THRESHOLD = -82.0  # IEEE Std 802.11-2020 (Revision of IEEE Std 802.11-2016), 17.3.10.6: CCA requirements

        self.associations = associations
        self.pos = pos
        self.walls_pos = walls_pos if walls_pos is not None else []
        self.walls = walls if walls is not None else self._calculate_walls_matrix()

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Scalar:
        pass

    def _calculate_walls_matrix(self) -> Array:
        """
        Converts a list of wall positions to a matrix counting the walls between each pair of nodes.

        Returns
        -------
        Array
            Matrix counting the walls between each pair of nodes.
        """

        def det(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

        def intersect(a, b, c, d):
            return det(a, b, c) * det(a, b, d) < 0 and det(a, c, d) * det(b, c, d) < 0

        nodes = list(self.associations.keys()) + list(chain.from_iterable(self.associations.values()))
        walls = np.zeros((len(nodes), len(nodes)), dtype=np.float32)

        for wall in self.walls_pos:
            for node_a, node_b in product(nodes, nodes):
                if node_a > node_b and intersect(self.pos[node_a], self.pos[node_b], wall[:2], wall[2:]):
                    walls[node_a, node_b] += 1
                    walls[node_b, node_a] += 1

        return walls

    def get_associations(self) -> Dict:
        return self.associations

    def plot(self, pos: Array, filename: str = None) -> None:
        colors = get_cmap(len(self.associations))
        ap_labels = string.ascii_uppercase

        _, ax = plt.subplots()

        for i, (ap, stations) in enumerate(self.associations.items()):
            ax.scatter(pos[ap, 0], pos[ap, 1], marker='x', color=colors[i])
            ax.scatter(pos[stations, 0], pos[stations, 1], marker='.', color=colors[i])
            ax.annotate(f'AP {ap_labels[i]}', (pos[ap, 0], pos[ap, 1] + 2), color=colors[i], va='bottom', ha='center')

            radius = np.max(np.sqrt(np.sum((pos[stations, :] - pos[ap, :]) ** 2, axis=-1)))
            circle = plt.Circle((pos[ap, 0], pos[ap, 1]), radius * 1.2, fill=False, linewidth=0.5)
            ax.add_patch(circle)

        # Plot walls
        for wall in self.walls_pos:
            ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color='black', linewidth=1)

        ax.set_axisbelow(True)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        ax.set_title('Location of nodes')
        ax.grid()

        if filename:
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()

    def is_cca_single_tx(self, pos: Array, tx_power: Array) -> bool:
        """
        Check if the scenario is a CSMA single transmission scenario, i.e., if there is only one transmission
        possible at a time due to the CCA threshold. **Note**: This function assumes that the scenario
        contains downlink transmissions only.

        Parameters
        ----------
        pos : Array
            Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
        tx_power : Array
            Transmission power of the nodes. Each entry corresponds to a node.

        Returns
        -------
        bool
            True if the scenario is a CSMA single transmission scenario, False otherwise.
        """

        ap_ids = np.array(list(self.associations.keys()))

        ap_pos = pos[ap_ids]
        ap_tx_power = tx_power[ap_ids]
        ap_walls = self.walls[ap_ids][:, ap_ids]

        distance = np.sqrt(np.sum((ap_pos[:, None, :] - ap_pos[None, ...]) ** 2, axis=-1))
        signal_power = ap_tx_power - path_loss(distance, ap_walls)
        signal_power = np.where(np.isnan(signal_power), np.inf, signal_power)

        return np.all(signal_power > self.CCA_THRESHOLD)

    def tx_matrix_to_action(self, tx_matrix: Array) -> list:
        """
        Convert a transmission matrix to a list of transmissions. Assumes downlink.

        Parameters
        ----------
        tx_matrix: Array
            Transmission matrix. Each entry corresponds to a node.

        Returns
        -------
        list
            A list, where each entry is either one element list of the AP->STA transmission or an empty one.
        """

        aps = list(self.associations.keys())
        action = [[] for _ in aps]

        for ap in aps:
            assert np.sum(tx_matrix[ap, :]) <= 1, 'Multiple transmissions at AP'
            for sta in self.associations[ap]:
                if tx_matrix[ap, sta]:
                    action[ap].append(sta)

        return action


class StaticScenario(Scenario):
    """
    Static scenario with fixed node positions, MCS, tx power, and noise standard deviation.
    The configuration of parallel transmissions is variable.

    Parameters
    ----------
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    mcs: int
        Modulation and coding scheme of the nodes. Each entry corresponds to a node.
    associations: Dict
        Dictionary of associations between access points and stations.
    default_tx_power: Scalar
        Transmission power of the nodes. Each entry corresponds to a node.
    sigma: Scalar
        Standard deviation of the additive white Gaussian noise.
    walls: Optional[Array]
        Matrix counting the walls between each pair of nodes.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to X and Y coordinates of a wall.
    tx_power_delta: Scalar
        Difference in transmission power between the tx power levels.
    """

    def __init__(
            self,
            pos: Array,
            mcs: int,
            associations: Dict,
            default_tx_power: Scalar = DEFAULT_TX_POWER,
            sigma: Scalar = DEFAULT_SIGMA,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None,
            tx_power_delta: Scalar = 6.0
    ) -> None:
        super().__init__(associations, pos, walls, walls_pos)

        self.pos = pos
        self.mcs = jnp.full(pos.shape[0], mcs, dtype=jnp.int32)
        self.tx_power = jnp.full(pos.shape[0], default_tx_power)
        self.tx_power_delta = tx_power_delta

        self.data_rate_fn = jax.jit(partial(
            network_data_rate,
            pos=self.pos,
            mcs=self.mcs,
            sigma=sigma,
            walls=self.walls
        ))

    def __call__(self, key: PRNGKey, tx: Array, tx_power: Optional[Array] = None) -> Scalar:
        if tx_power is None:
            tx_power = jnp.zeros_like(self.tx_power)

        return self.data_rate_fn(key, tx, tx_power=self.tx_power - self.tx_power_delta * tx_power)

    def plot(self, filename: str = None) -> None:
        super().plot(self.pos, filename)

    def is_cca_single_tx(self) -> bool:
        return super().is_cca_single_tx(self.pos, self.tx_power)
