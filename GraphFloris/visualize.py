import dgl
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge


def visualize_wind_farm(g: dgl.DGLGraph,
                        min_distance: float,
                        angle_threshold: float,
                        wind_direction: float,
                        wind_speed: float,
                        x_grid_size: float,
                        y_grid_size: float,
                        dir_mark_margin=0.01,
                        dir_mark_len=0.03,
                        arrow_size=0.01,
                        viz_eps=0.02,
                        dpi=250,
                        return_fig=False,
                        label_size=15,
                        tick_size=12,
                        annotation_size=12,
                        show_color_bar_label=False,
                        title_size=20,
                        show_title=False,
                        edge_width=8,
                        legend_size=15):
    # Figure drawing parameters
    influential_region_zorder = 0
    min_distance_region_zorder = 1
    edge_zorder = 2
    turbine_zorder = 3

    scatter_line_width = 2.0
    scatter_ball_size = 200
    x_limit = x_grid_size
    y_limit = y_grid_size

    fig, (ax, ax_colorbar) = plt.subplots(1, 2,
                                          figsize=(10.5, 10),
                                          gridspec_kw={'width_ratios': [10, 0.5]},
                                          dpi=dpi)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    ax.xaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='minor')
    ax.set_xlim([-x_limit * viz_eps, x_limit * (1 + viz_eps)])
    ax.set_ylim([-y_limit * viz_eps, y_limit * (1 + viz_eps)])
    ax.tick_params(labelsize=tick_size)

    # Draw turbines color-coded with powers
    x, y = g.ndata['x'], g.ndata['y']
    cmap = matplotlib.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    turbine_scatter = ax.scatter(x, y,
                                 label='Turbine',
                                 cmap=cmap,
                                 c=norm(g.ndata['power']),
                                 edgecolors='#bbbbbb',
                                 linewidths=scatter_line_width,
                                 s=scatter_ball_size,
                                 zorder=turbine_zorder)

    # Draw interaction edges
    xys = torch.cat([x, y], dim=-1).squeeze().numpy()
    pos_dict = {}
    for i, xy in enumerate(xys):
        pos_dict[i] = xy
    nxg = dgl.to_networkx(g)
    nx.draw_networkx_edges(nxg,
                           edge_color='grey',
                           ax=ax,
                           pos=pos_dict)

    ax.set_xlabel("Wind farm X size (m)", fontsize=label_size)
    ax.set_ylabel("Wind farm Y size (m)", fontsize=label_size)

    cbar = fig.colorbar(turbine_scatter,
                        cax=ax_colorbar,
                        ticks=np.arange(0.0, 1.1, 0.05))

    if show_color_bar_label:
        cbar.set_label('Power', fontsize=label_size)
    cbar.set_clim(0.0, 1.0)
    cbar.ax.tick_params(labelsize=tick_size)

    for i, (x, y) in enumerate(zip(g.ndata['x'], g.ndata['y'])):
        # Add min-distance circle
        circle = plt.Circle((x, y),
                            min_distance / 2,
                            hatch='....',
                            facecolor='white',
                            edgecolor='#ffa45c',
                            alpha=1.0,
                            zorder=min_distance_region_zorder)

        # Add annotation for depicting turbine index and the corresponding power
        ax.annotate("T{}".format(i + 1),
                    xy=(x, y + y_limit * 0.025),
                    fontsize=annotation_size,
                    horizontalalignment="center")
        # ax.annotate("Power : {:.2f}".format(index_power_dict[node.index]),
        #             (x, y - y_limit * viz_eps))
        ax.add_artist(circle)

        # Add influential cones
        wedge = Wedge(
            (x, y),
            np.sqrt(x_grid_size**2 + y_grid_size**2),  # radius
            270 - wind_direction - angle_threshold,  # from theta 1 (in degrees) # FLORIS 1.4 measures 270 degree as left
            270 - wind_direction + angle_threshold,  # to theta 2 (in degrees)
            color='g', alpha=0.05,
            zorder=influential_region_zorder)
        ax.add_patch(wedge)

    # Draw directional mark
    dir_mark_center_x = x_limit * (1 - 2 * dir_mark_margin - dir_mark_len)
    dir_mark_center_y = y_limit * (1 - 2 * dir_mark_margin - dir_mark_len)

    marker_len = x_limit * dir_mark_len

    # WEST
    dir_mark_west_x = dir_mark_center_x - marker_len
    dir_mark_west_y = dir_mark_center_y

    # EAST
    dir_mark_east_x = dir_mark_center_x + marker_len
    dir_mark_east_y = dir_mark_center_y

    # NORTH
    dir_mark_north_x = dir_mark_center_x
    dir_mark_north_y = dir_mark_center_y + marker_len

    # SOUTH
    dir_mark_south_x = dir_mark_center_x
    dir_mark_south_y = dir_mark_center_y - marker_len

    # Visualize wind direction

    # Compass
    compass_color = 'k'
    wd_color = 'orange'
    arrow_size = x_limit * arrow_size

    # WEST -> EAST
    w2e = Line2D((dir_mark_west_x, dir_mark_east_x),
                 (dir_mark_west_y, dir_mark_east_y),
                 alpha=0.5, c=compass_color)

    ax.add_line(w2e)

    # NORTH -> SOUTH
    n2s = Line2D((dir_mark_north_x, dir_mark_south_x),
                 (dir_mark_north_y, dir_mark_south_y),
                 alpha=0.5, c=compass_color)

    ax.add_line(n2s)

    # NORTH -> WEST
    n2w = Line2D((dir_mark_north_x, dir_mark_west_x),
                 (dir_mark_north_y, dir_mark_west_y),
                 alpha=0.5, c=compass_color)

    ax.add_line(n2w)

    # draw wind direction arrow
    wind_dir_rad = np.radians(wind_direction-270)
    sin = np.sin(wind_dir_rad)
    cos = np.cos(wind_dir_rad)
    wind_start_x = dir_mark_center_x - marker_len * cos  # tail
    wind_start_y = dir_mark_center_y + marker_len * sin

    wind_end_x = dir_mark_center_x + marker_len * cos  # arrow
    wind_end_y = dir_mark_center_y - marker_len * sin

    ax.arrow(wind_start_x, wind_start_y,
             wind_end_x - wind_start_x, wind_end_y - wind_start_y,
             linewidth=2,
             head_width=arrow_size, head_length=arrow_size,
             fc=wd_color, ec=wd_color, length_includes_head=True,
             zorder=edge_zorder)

    ax.annotate("Wind direction : {}$^\circ$".format(wind_direction),
                (dir_mark_west_x - marker_len * 7.0,
                 dir_mark_center_y + marker_len * 0.5))

    ax.annotate("Wind speed : {} m/s".format(wind_speed),
                (dir_mark_west_x - marker_len * 7.0,
                 dir_mark_center_y - marker_len * 0.5))

    handles, labels = ax.get_legend_handles_labels()
    handles += [wedge, circle]
    labels += ["influential region", "min. distance"]
    ax.legend(handles, labels, prop={'size': legend_size})

# if show_title:
    #     fig.suptitle('Wind Farm Layout', fontsize=title_size)
    #
    # if return_fig:
    #     return fig, ax
