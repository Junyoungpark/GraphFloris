import numpy as np

from GraphFloris.WindFarm import WindFarm


def prepare_data(farm, n_samples, wind_speed_range, wind_direction_range, num_turbine_list):
    nts = np.random.choice(num_turbine_list,
                           size=n_samples)
    wds = np.random.uniform(low=wind_direction_range[0],
                            high=wind_direction_range[1],
                            size=n_samples)
    wss = np.random.uniform(low=wind_speed_range[0],
                            high=wind_speed_range[1],
                            size=n_samples)

    gs = []
    for n, wd, ws in zip(nts, wds, wss):
        farm.sample_layout(n)
        farm.update_graph(wind_speed=ws, wind_direction=wd)
        gs.append(farm.observe())
    return gs


if __name__ == '__main__':
    num_turbines = [5, 10, 15, 20]
    x_grid_size, y_grid_size = 2000, 2000
    wd_min, wd_max = 0, 360  # degree; 0: North wind (North -> South) 90: East wind (East -> West)
    ws_min, ws_max = 6.0, 15.0  # m/sec

    farm = WindFarm(5, x_grid_size, y_grid_size)

    n_train = 900
    n_val = 100

    train_gs = prepare_data(farm, n_train,
                            wind_speed_range=[ws_min, ws_max],
                            wind_direction_range=[wd_min, wd_max],
                            num_turbine_list=num_turbines)
