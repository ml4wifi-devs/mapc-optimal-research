from itertools import chain

import numpy as np
from chex import Array

from mapc_mab.mapc_agent import MapcAgent


class RandomMapcAgentFactory:
    def __init__(self, associations: dict[int, list[int]], seed: int = 42) -> None:
        self.associations = associations
        np.random.seed(seed)

    def create_mapc_agent(self) -> MapcAgent:
        return RandomMapcAgent(self.associations)


class RandomMapcAgent(MapcAgent):
    def __init__(self, associations: dict[int, list[int]]) -> None:
        self.associations = {ap: np.array(stations) for ap, stations in associations.items()}
        self.access_points = np.array(list(associations.keys()))
        self.n_nodes = len(self.access_points) + len(list(chain.from_iterable(associations.values())))

    def sample(self, _) -> tuple[Array, Array]:
        sharing_ap = np.random.choice(self.access_points).item()
        designated_station = np.random.choice(self.associations[sharing_ap]).item()

        tx_matrix = np.zeros((self.n_nodes, self.n_nodes), dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1
        tx_power = np.zeros(self.n_nodes, dtype=np.int32)

        return tx_matrix, tx_power
