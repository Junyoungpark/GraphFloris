from typing import List

import dgl
import matplotlib.pyplot as plt
import numpy as np
import torch
from floris.tools.floris_interface import FlorisInterface

from GraphFloris.config import EXAMPLE_FARM
from GraphFloris.gen_graph import get_node_only_graph, update_edges
from GraphFloris.layout_sampling import sequential_sampling
from GraphFloris.visualize import visualize_wind_farm


class WindFarm:

    def __init__(self,
                 num_turbines: int,
                 x_grid_size: float = 3000,  # the x size of windfarm is by default 3000m
                 y_grid_size: float = 3000,  # the y size of windfarm is by default 3000m
                 angle_threshold: float = 90.0,  # the angle threshold (degree)
                 min_distance_factor: float = 2.0,  # minimal safety distance factor between two turbines
                 dist_cutoff_factor: float = 50.0):  # maximal influential distance factor between two turbines
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
        self.wind_speed = None  # free flow wind speed (m/sec)
        self.wind_direction = None  # direction of wind (degree)

        # comments on wind direction
        # 0 (360) = North -> South
        # 90 = East -> West
        # 180 = South -> North
        # 270 = West -> East

        self.g = None
        self.sample_layout(num_turbines, x_grid_size, y_grid_size)

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
        self.num_turbines = num_turbines
        self.g = get_node_only_graph(self.xs, self.ys)

    def set_power(self, g, wind_speed: float, wind_direction: float):
        # prepare normalizer
        self._one_turbine_farm.reinitialize_flow_field(wind_speed=wind_speed,
                                                       wind_direction=wind_direction)
        self._one_turbine_farm.calculate_wake()
        one_farm_power = self._one_turbine_farm.get_farm_power()  # scalar

        # compute wind turbine powers
        self._farm.reinitialize_flow_field(wind_speed=wind_speed,
                                           wind_direction=wind_direction)
        self._farm.calculate_wake()
        powers = self._farm.get_turbine_power()
        norm_powers = np.array(powers) / one_farm_power
        g.ndata['power'] = torch.tensor(norm_powers).view(-1, 1).float()

    def update_graph(self,
                     wind_speed: float,
                     wind_direction: float,
                     xs: List[float] = None,
                     ys: List[float] = None,
                     angle_threshold: float = None,
                     dist_cutoff_factor: float = None):
        # update graph
        if xs is not None and ys is not None:  # construct graph
            self.g = get_node_only_graph(xs, ys)
            self.xs, self.ys = xs, ys
            self.num_turbines = len(xs)
        if wind_direction is not None:
            ag_th = self.angle_threshold if angle_threshold is None else angle_threshold
            cutoff_dist = self.cutoff_dist if dist_cutoff_factor is None else dist_cutoff_factor * self.turbine_diameter
            update_edges(self.g, wind_direction, ag_th, cutoff_dist)
        if wind_speed is not None:
            self.set_power(self.g, wind_speed, wind_direction)

        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

    def observe(self):
        g = dgl.DGLGraph(self.g)  # copy structure of wind farm DGLGraph

        # setup internal node features and edge features
        for k in self.g.ndata.keys():
            g.ndata[k] = self.g.ndata[k]

        for k in self.g.edata.keys():
            g.edata[k] = self.g.edata[k]

        # setup node/edge attributes
        n = g.number_of_nodes()
        g.ndata['wind_speed'] = torch.ones(n, 1) * self.wind_speed

        # setup features for regression models
        # this selection of node and edge features was investigated in the paper
        # 'Physics-Induced Graph Neural Network: An Application to wind-farm power prediction'
        # https://www.sciencedirect.com/science/article/pii/S0360544219315555

        # node feature
        g.ndata['feat'] = torch.ones(n, 1) * self.wind_speed

        # edge feature
        ef = torch.cat([g.edata['down_stream_dist'], g.edata['radial_dist']], dim=-1)
        g.edata['feat'] = ef

        # global feature
        u = torch.ones(1, 1) * self.wind_speed

        # regression target (usually simulated power of turbines) are already assigned to the graph
        # g.ndata['power']

        return g, u

    def update_config(self,
                      angle_threshold: float = None,
                      min_distance_factor: float = None,
                      dist_cutoff_factor: float = None):
        if angle_threshold is not None:
            self.angle_threshold = angle_threshold
        if min_distance_factor is not None:
            self.min_distance = min_distance_factor * self.turbine_diameter
        if dist_cutoff_factor is not None:
            self.cutoff_dist = dist_cutoff_factor * self.turbine_diameter

    def visualize_farm(self, **viz_kwargs):
        assert self.g is not None, "construct graph first! you can construct wind farm graph with 'update_graph'"
        visualize_wind_farm(g=self.g,
                            min_distance=self.min_distance,
                            angle_threshold=self.angle_threshold,
                            wind_direction=self.wind_direction,
                            wind_speed=self.wind_speed,
                            x_grid_size=self.x_grid_size,
                            y_grid_size=self.y_grid_size,
                            **viz_kwargs)
        plt.show()


if __name__ == '__main__':
    farm = WindFarm(30, 3000, 3000)
    farm.update_graph(12, 90)
    farm.visualize_farm()
    g, u = farm.observe()

    print(g, u)
