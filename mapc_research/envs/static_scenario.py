from functools import partial
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from chex import Array, Scalar, PRNGKey
from mapc_sim.sim import network_data_rate
from mapc_sim.constants import DEFAULT_TX_POWER, DEFAULT_SIGMA, DATA_RATES, TAU

from mapc_research.envs.scenario import Scenario


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
    """

    def __init__(
            self,
            pos: Array,
            mcs: int,
            associations: Dict,
            n_steps: int,
            default_tx_power: Scalar = DEFAULT_TX_POWER,
            sigma: Scalar = DEFAULT_SIGMA,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None,
            tx_power_delta: Scalar = 3.0,
            str_repr: str = ""
    ) -> None:
        super().__init__(associations, pos, walls, walls_pos)

        self.pos = pos
        self.mcs = mcs
        self.tx_power = jnp.full(pos.shape[0], default_tx_power)
        self.tx_power_delta = tx_power_delta
        self.n_steps = n_steps
        self.sigma = sigma

        self.data_rate_fn = jax.jit(partial(
            network_data_rate,
            pos=self.pos,
            mcs=None,
            sigma=self.sigma,
            walls=self.walls
        ))
        self.normalize_reward = DATA_RATES[-1]

        self.str_repr = "static_" + str_repr if str_repr else "static"
    
    def __str__(self):
        return self.str_repr

    def __call__(self, key: PRNGKey, tx: Array, tx_power: Optional[Array] = None) -> tuple[Scalar, Scalar]:
        if tx_power is None:
            tx_power = jnp.zeros_like(self.tx_power)

        thr = self.data_rate_fn(key, tx, tx_power=self.tx_power - self.tx_power_delta * tx_power)
        reward = thr / self.normalize_reward
        return thr, reward

    def split_scenario(self) -> list[tuple['StaticScenario', float]]:
        return [(self, self.n_steps * TAU)]

    def plot(self, filename: str = None) -> None:
        super().plot(self.pos, filename)

    def is_cca_single_tx(self) -> bool:
        return super().is_cca_single_tx(self.pos, self.tx_power)
