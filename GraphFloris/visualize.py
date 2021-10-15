import dgl
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from matplotlib.ticker import FuncFormatter


def visualize_wind_farm(g: dgl.DGLGraph,
                        min_distance: float,
                        cutoff_dist: float,
                        angle_threshold: float,
                        influence_radius: float,
                        wind_direction: float,
                        wind_speed: float,
                        x_grid_size: float,
                        y_grid_size: float,
                        dir_mark_margin=0.01,
                        dir_mark_len=0.03,
                        arrow_size=0.01,
                        viz_eps=0.02,
                        dpi=250,
                        label_size=15,
                        tick_size=12,
                        annotation_size=12,
                        show_color_bar_label=True,
                        edge_width=2.0,
                        legend_size=15,
                        draw_wedges=True,
                        highlighted_turbine=None,
                        img=None,
                        details=True,
                        annotate=True):
    # Figure drawing parameters
    influential_region_zorder = -3
    min_distance_region_zorder = -2
    img_zorder = -1
    edge_zorder = 2
    turbine_zorder = 3

    scatter_line_width = 2.0
    scatter_ball_size = 200 if img is None else 100
    x_limit = x_grid_size
    y_limit = y_grid_size

    fig = plt.figure(figsize=(10.5, 10), dpi=dpi)
    ax = fig.gca()
    ax_colorbar = fig.add_axes([.90, .2, .015, .6])  # create a colorbar axes
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.subplots_adjust(right=.89)  # add space so the colorbar doesn't overlap the plot

    ax.axis('scaled')
    ax.xaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='minor')
    ax.yaxis.grid(True, which='major')
    ax.yaxis.grid(True, which='minor')
    if img is None:
        ax.set_xlim([-x_limit * viz_eps, x_limit * (1 + viz_eps)])
        ax.set_ylim([-y_limit * viz_eps, y_limit * (1 + viz_eps)])
    else:
        ax.set_xlim(img['extent'][:2])
        ax.set_ylim(img['extent'][2:])
    ax.tick_params(labelsize=tick_size)


    # Draw turbines color-coded with powers
    x, y = g.ndata['x'], g.ndata['y']
    cmap = matplotlib.cm.Blues
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    edgecolors = ['#bbbbbb'] * len(x)
    linewidths = [scatter_line_width] * len(x)
    if highlighted_turbine is not None:
        edgecolors[highlighted_turbine] = 'g'
        linewidths[highlighted_turbine] *= 1.5

    if img is not None:
        im = plt.imread(img['path'])
        ax.imshow(im, zorder=img_zorder, extent=img['extent'], alpha=1)
    turbine_scatter = ax.scatter(x, y,
                                 label='Turbine',
                                 cmap=cmap,
                                 c=norm(g.ndata['power']),
                                 edgecolors=edgecolors,
                                 linewidths=linewidths,
                                 s=scatter_ball_size,
                                 zorder=turbine_zorder)
    index_power_dict = g.ndata['power'].squeeze().tolist()

    # Draw interaction edges
    xys = torch.cat([x, y], dim=-1).squeeze().numpy()
    pos_dict = {}
    for i, xy in enumerate(xys):
        pos_dict[i] = xy

    if details and 'weight' in g.edata.keys():
        # For draw_networkx_edge_labels, we need 2-tuple -> we need get_edge_attributes on a DiGraph not MultiDiGraph
        nxg = nx.DiGraph(dgl.to_networkx(g, edge_attrs=['weight']))
        edge_data = nx.get_edge_attributes(nxg, 'weight')
        # Display edge weight
        edge_labels = {k: f"{v.tolist():.0%}" for k, v in edge_data.items() }
        nx.draw_networkx_edge_labels(nxg, 
                                     ax=ax,
                                     edge_labels=edge_labels,
                                     label_pos=3.2/5,
                                     font_size=8,
                                     font_color='green',
                                     font_weight='bold',
                                     pos=pos_dict)
        # Edge width based on edge weight
        edges = list(nxg.edges())
        edge_width = { edges.index(k): edge_width * (0.5 + v) for k, v in edge_data.items() }
    else:
        nxg = dgl.to_networkx(g)

    if details:
        nx.draw_networkx_edges(nxg,
                            edge_color='grey',
                            ax=ax,
                            width=edge_width,
                            pos=pos_dict)

    ax.set_xlabel("Wind farm X size (m)", fontsize=label_size)
    ax.set_ylabel("Wind farm Y size (m)", fontsize=label_size)

    # Format ticks to % | Ref.: https://stackoverflow.com/a/68106284
    fmt = lambda x, pos: '{:.0%}'.format(x)
    cbar = fig.colorbar(turbine_scatter,
                        cax=ax_colorbar,
                        format=FuncFormatter(fmt),
                        ticks=np.arange(0.0, 1.1, 0.05))

    if show_color_bar_label:
        cbar.set_label('Power', fontsize=label_size)
    cbar.mappable.set_clim(0.0, 1.0)
    cbar.ax.tick_params(labelsize=tick_size)

    circle, wedge = None, None
    if annotate:
        for i, (x, y) in enumerate(zip(g.ndata['x'], g.ndata['y'])):
            # Add annotation for depicting turbine index and the corresponding power
            ax.annotate("T{}".format(i + 1),
                        xy=(x, y + y_limit * 0.025),
                        fontsize=annotation_size,
                        horizontalalignment="center")
            ax.annotate("P: {:.0%}".format(index_power_dict[i]),
                        (x, y - y_limit * 0.03),
                        horizontalalignment="center")

    if details:
        for i, (x, y) in enumerate(zip(g.ndata['x'], g.ndata['y'])):
            # Add min-distance circle
            circle = plt.Circle((x, y),
                                min_distance / 2,
                                hatch='....',
                                facecolor='white',
                                edgecolor='#ffa45c',
                                alpha=1.0,
                                zorder=min_distance_region_zorder)
            ax.add_artist(circle)

            if draw_wedges:
                # Add influential cones
                wedge = Wedge(
                    (float(x), float(y)),
                    cutoff_dist, # np.sqrt(x_grid_size ** 2 + y_grid_size ** 2),  # radius
                    270 - wind_direction - angle_threshold,
                    # from theta 1 (in degrees) # FLORIS 1.4 measures 270 degree as left
                    270 - wind_direction + angle_threshold,  # to theta 2 (in degrees)
                    color='g', alpha=0.05,
                    linewidth=0,
                    zorder=influential_region_zorder)
                ax.add_patch(wedge)
            
                # Add influential radius
                wind_dir_rad = np.radians(wind_direction)
                sin = np.sin(wind_dir_rad)
                cos = np.cos(wind_dir_rad)
                radius_on_turbine_dx = influence_radius * cos
                radius_on_turbine_dy = influence_radius * sin

                d = influence_radius / np.tan(np.radians(angle_threshold))
                triangle_opening_dx = d * sin
                triangle_opening_dy = d * cos

                wedge_radius = plt.Polygon([[x - radius_on_turbine_dx, y + radius_on_turbine_dy],
                                            [x - radius_on_turbine_dx - triangle_opening_dx, y + radius_on_turbine_dy - triangle_opening_dy],
                                            [x, y],
                                            [x + radius_on_turbine_dx - triangle_opening_dx, y - radius_on_turbine_dy - triangle_opening_dy], 
                                            [x + radius_on_turbine_dx, y - radius_on_turbine_dy]],
                                        color='g',
                                        linewidth=0,
                                        alpha=0.05,
                                        zorder=influential_region_zorder,
                                        closed=True,
                                        fill=True)
                ax.add_patch(wedge_radius)        

    # Draw directional mark
    dir_mark_center_x = x_limit * (1 - 2 * dir_mark_margin - dir_mark_len)
    dir_mark_center_y = y_limit * (1 - (1 - 2 * dir_mark_margin - dir_mark_len))

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
    wind_dir_rad = np.radians(wind_direction - 270)
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

    # Add background
    wedge_radius = plt.Polygon([
                                    [dir_mark_west_x - marker_len * 7.5, dir_mark_center_y + marker_len * 1.5],
                                    [x_limit, dir_mark_center_y + marker_len * 1.5],
                                    [x_limit, 0],
                                    [dir_mark_west_x - marker_len * 7.5, 0]
                                ],
                                color='white',
                                linewidth=0,
                                alpha=0.5,
                                zorder=-0.5,
                                closed=True,
                                fill=True)
    ax.add_patch(wedge_radius)    

    handles, labels = ax.get_legend_handles_labels()
    handles += [circle, wedge, wedge_radius] if wedge is not None else [circle]
    labels += ["min. distance", "influential region"] if wedge is not None else ["min. distance"]
    ax.legend(handles, labels, prop={'size': legend_size})

# if show_title:
#     fig.suptitle('Wind Farm Layout', fontsize=title_size)
#
# if return_fig:
#     return fig, ax
