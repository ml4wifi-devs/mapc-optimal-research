from functools import partial
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
from chex import Array, Scalar, PRNGKey
from mapc_sim.sim import network_data_rate
from mapc_sim.constants import DEFAULT_TX_POWER, DEFAULT_SIGMA, DATA_RATES, TAU
from mapc_sim.utils import default_path_loss

from mapc_research.envs.scenario import Scenario


class StaticScenario(Scenario):
    """
    Static scenario with fixed node positions and associations.

    Parameters
    ----------
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    associations: Dict
        Dictionary of associations between access points and stations.
    n_steps: int
        Number of steps in the simulation.
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
    path_loss_fn: Callable
        A function that calculates the path loss between two nodes. The function signature should be
        `path_loss_fn(distance: Array, walls: Array) -> Array`, where `distance` is the matrix of distances
        between nodes and `walls` is the adjacency matrix of walls. By default, the simulator uses the
        residential TGax path loss model.
    str_repr: str
        String representation of the scenario.
    """

    def __init__(
            self,
            pos: Array,
            associations: Dict,
            n_steps: int = float('inf'),
            default_tx_power: Scalar = DEFAULT_TX_POWER,
            sigma: Scalar = DEFAULT_SIGMA,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None,
            tx_power_delta: Scalar = 3.0,
            path_loss_fn: Callable = default_path_loss,
            str_repr: str = ""
    ) -> None:
        self.str_repr = "static_" + str_repr if str_repr else "static"
        super().__init__(associations, pos, walls, walls_pos, path_loss_fn, self.str_repr)

        self.pos = pos
        self.tx_power = jnp.full(pos.shape[0], default_tx_power)
        self.tx_power_delta = tx_power_delta
        self.n_steps = n_steps
        self.sigma = sigma

        self.data_rate_fn = jax.jit(partial(
            network_data_rate,
            pos=self.pos,
            mcs=None,
            sigma=self.sigma,
            walls=self.walls,
            path_loss_fn=self.path_loss_fn
        ))
        self.normalize_reward = DATA_RATES[-1]

    def __call__(self, key: PRNGKey, tx: Array, tx_power: Optional[Array] = None) -> tuple[Scalar, Scalar]:
        if tx_power is None:
            tx_power = jnp.zeros_like(self.tx_power)

        thr = self.data_rate_fn(key, tx, tx_power=self.tx_power - self.tx_power_delta * tx_power)
        reward = thr / self.normalize_reward
        return thr, reward

    def split_scenario(self) -> list[tuple['StaticScenario', float]]:
        return [(self, self.n_steps * TAU)]

    def plot(self, filename: str = None, label_size: int = 10, show_circles: bool = True) -> None:
        super().plot(self.pos, filename, label_size, show_circles)

    def is_cca_single_tx(self) -> bool:
        return super().is_cca_single_tx(self.pos, self.tx_power)
