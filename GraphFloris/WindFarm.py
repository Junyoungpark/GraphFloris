from typing import List

import numpy as np
import torch
from floris.tools.floris_interface import FlorisInterface

from GraphFloris.config import EXAMPLE_FARM
from GraphFloris.gen_graph import get_node_only_graph, update_edges
from GraphFloris.layout_sampling import sequential_sampling


class WindFarm:

    def __init__(self,
                 num_turbines: int,
                 x_grid_size: float,
                 y_grid_size: float,
                 angle_threshold: float = 90.0,
                 min_distance_factor: float = 5.0,
                 dist_cutoff_factor: float = 50.0):
        self.num_turbines = num_turbines
        self.x_grid_size = x_grid_size
        self.y_grid_size = y_grid_size

        self.angle_threshold = angle_threshold
        self.turbine_diameter = EXAMPLE_FARM['turbine']['properties']['rotor_diameter']
        self.min_distance = min_distance_factor * self.turbine_diameter
        self.cutoff_dist = dist_cutoff_factor * self.turbine_diameter
        self._one_turbine_farm = FlorisInterface(input_dict=EXAMPLE_FARM)
        self._farm = FlorisInterface(input_dict=EXAMPLE_FARM)

        self.xs = None  # will be determined
        self.ys = None  # will be determined

        self.sample_layout(num_turbines, x_grid_size, y_grid_size)
        self.g = get_node_only_graph(self.xs, self.ys)
        self._built = False

    def sample_layout(self,
                      num_turbines: int,
                      x_grid_size: float = None,
                      y_grid_size: float = None):
        num_turbines = self.num_turbines if num_turbines is None else num_turbines
        x_grid_size = self.x_grid_size if x_grid_size is None else x_grid_size
        y_grid_size = self.y_grid_size if y_grid_size is None else y_grid_size
        self.xs, self.ys = sequential_sampling(x_grid_size=x_grid_size,
                                               y_grid_size=y_grid_size,
                                               min_dist=self.min_distance,
                                               num_turbines=num_turbines)
        self._farm.reinitialize_flow_field(layout_array=[self.xs, self.ys])
        self._built = False

    def set_power(self, g, wind_speed: float):
        # prepare normalizer
        self._one_turbine_farm.reinitialize_flow_field(wind_speed=wind_speed)
        self._one_turbine_farm.calculate_wake()
        one_farm_power = self._one_turbine_farm.get_farm_power()  # scalar

        # compute wind turbine powers
        self._farm.reinitialize_flow_field(wind_speed=wind_speed)
        self._farm.calculate_wake()
        powers = self._farm.get_turbine_power()
        norm_powers = np.array(powers) / one_farm_power
        g.ndata['power'] = torch.tensor(norm_powers).view(-1, 1).float()

    def update_graph(self,
                     wind_speed: float,
                     wind_direction: float,
                     xs: List[float] = None,
                     ys: List[float] = None):
        # update graph
        if xs is not None and ys is not None:  # construct graph
            self.g = get_node_only_graph(xs, ys)
        if wind_direction is not None:
            update_edges(self.g, wind_direction, self.angle_threshold, self.cutoff_dist)
        if wind_speed is not None:
            self.set_power(self.g, wind_speed)

        self._built = True

    def observe(self):
        pass


if __name__ == '__main__':
    farm = WindFarm(10, 3000, 3000)
    farm.update_graph(12, 0)
