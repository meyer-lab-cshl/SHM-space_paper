import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

import seaborn as sn
from collections import defaultdict
from collections import Counter
import networkx as nx

color_red = "#D21F23"
color_blue = "#81D4FA"
color_yellow = "#FFC107"
color_gray = "#D9D9D9"
color_violet = "#A495AC"

############################ Graph for the simulation  ############################

def plot_mutational_paths(paths_array, v1, v2, stat_freq_norm, sequence_to_index, sims, mut_stop=False, time_stop=False,
                          base_size=3, base_alpha=0.3, threshold=1,
                          variable_node_size=True, node_alpha=1.0):

    """
    Draw simulated paths in diffusion coordinates, colored by log stationary frequency.
    :param paths_array: sequence of sequences: simulated paths of nucleotide sequences.
    :param v1: array-like: first diffusion coordinate indexed by sequence.
    :param v2: array-like: second diffusion coordinate indexed by sequence.
    :param stat_freq_norm: np.ndarray: stationary frequencies indexed by sequence.
    :param sequence_to_index: dict[str, int]: sequence to coordinate index.
    :param sims: int: number of simulations, used in count adjustment and title.
    :param mut_stop: int or False: number of mutation events; takes precedence when truthy.
    :param time_stop: float or False: end time; in simulation functions, enables a time stopping rule when truthy.
    :param base_size: float: baseline node size.
    :param base_alpha: float: edge opacity multiplier per traversal, capped at one.
    :param threshold: int: minimum visit count used for display; exact comparison depends on the plot.
    :param variable_node_size: bool: scale node size with visit count.
    :param node_alpha: float: node opacity.
    :return: None: creates a Matplotlib figure. Starting sequences are labeled; node counts and edge opacity reflect trajectory visits.
    """

    G = nx.Graph()
    node_counter = Counter()
    edge_counter = defaultdict(int)

    starting_seqs = list(set(row[0] for row in paths_array))

    # Count node and edge frequencies from paths
    for row in paths_array:
        for i, seq in enumerate(row):
            node_counter[seq] += 1
            if i == 0:
                node_counter[seq] -= 1
            if i < len(row) - 1:
                edge_counter[tuple(sorted([row[i], row[i + 1]]))] += 1

    # Build node attributes
    for seq, idx in sequence_to_index.items():
        count = node_counter.get(seq, 0)
        if seq in starting_seqs:
            count -= sims + 1
        if count < threshold and seq not in starting_seqs:
            continue
        pos = (v1[idx].real, v2[idx].real)
        color = np.log(stat_freq_norm[idx].real) if count != 0 else None  # None for unvisited
        G.add_node(seq, pos=pos, count=count, color=color)

    # Add edges
    for (a, b), count in edge_counter.items():
        if a in G and b in G:
            G.add_edge(a, b, alpha=min(1.0, base_alpha * count))

    # Prepare node data
    pos_dict = nx.get_node_attributes(G, "pos")

    # Precompute properties
    active_pos = []
    active_color = []
    active_size = []

    inactive_pos = []
    inactive_size = []

    for node in G.nodes:
        x, y = pos_dict[node]
        count = G.nodes[node]['count']
        color = G.nodes[node]['color']
        size = base_size + (count * 10) if variable_node_size else base_size

        if color is not None:
            active_pos.append((x, y))
            active_color.append(color)
            active_size.append(size)
        else:
            inactive_pos.append((x, y))
            inactive_size.append(size)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Draw edges
    for u, v, data in G.edges(data=True):
        x0, y0 = pos_dict[u]
        x1, y1 = pos_dict[v]
        ax.plot([x0, x1], [y0, y1], color='black', alpha=data['alpha'], linewidth=0.2)

    # Plot unvisited (gray) nodes first
    if inactive_pos:
        ax.scatter(
            [p[0] for p in inactive_pos],
            [p[1] for p in inactive_pos],
            c='Gainsboro',
            s=inactive_size,
            alpha=node_alpha
        )

    # Plot visited (colored) nodes on top
    if active_pos:
        ax.scatter(
            [p[0] for p in active_pos],
            [p[1] for p in active_pos],
            c=active_color,
            s=active_size,
            alpha=node_alpha,
            cmap='YlOrRd'
        )

    # Force layout update for accurate axis limits
    fig.canvas.draw()

    # Add labels to starting sequences
    y_min, y_max = ax.get_ylim()
    y_offset = 0.01 * (y_max - y_min)
    for seq in starting_seqs:
        if seq in G.nodes:
            x, y = pos_dict[seq]
            ax.text(x, y + y_offset, seq, fontsize=8, ha='center', va='bottom',
                    color='#81D4FA', alpha=1, weight='bold')

    # Colorbar
    norm = plt.Normalize(np.log(stat_freq_norm.real).min(), np.log(stat_freq_norm.real).max())
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("ln pi", fontsize=12)

    ax.set_xlabel("Log stationary distribution", fontsize=12)
    ax.set_ylabel("Diffusion axis 1", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if mut_stop:
        plt.title(f'{starting_seqs[0]}: {sims} simulations, stop after {mut_stop}-th mutation')
    elif time_stop:
        plt.title(f'{starting_seqs[0]}: {sims} simulations, stop after time = {time_stop}')
    else:
        plt.title(f'{starting_seqs[0]}: {sims} simulations')
    plt.subplots_adjust(top=0.95, bottom=0.1)
    plt.tight_layout()

############################ Scatterplot how CDRs explore SHM space  ############################

def plot_kmer_occurrence(x, y, sequences, count_maps, colors, starting_seqs_list,
                         labels=[], threshold=10, ax=None):
    """
    Plot k-mer occurrences by group and highlight starting sequences.
    :param x: array-like: first sample or x coordinates, according to the function.
    :param y: array-like: second sample or y coordinates, according to the function.
    :param sequences: sequence[str]: ordered sequences; indices must align with associated vectors or matrices.
    :param count_maps: sequence[dict]: visit counts per sequence for each group.
    :param colors: sequence: Matplotlib colors for groups.
    :param starting_seqs_list: sequence of sequences: starting sequences per group.
    :param labels: sequence or dict: group labels for occurrence plots, or amino acid label mapping for dumbbell plots.
    :param threshold: int: minimum visit count used for display; exact comparison depends on the plot.
    :param ax: matplotlib.axes.Axes or None: axes to draw on; None uses current axes.
    :return: None: draws on ax or the current axes. A k-mer is assigned to the first group exceeding threshold; only groups paired with labels are drawn in color.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Use current axes if none provided
    if ax is None:
        ax = plt.gca()

    # Convert input to arrays
    sequences = np.array(sequences)
    x = np.array(x)
    y = np.array(y)

    # Combine all starting sequences into a single set
    all_starting_seqs = set(seq for group in starting_seqs_list for seq in group)

    # Data containers
    gray_x, gray_y, gray_sizes = [], [], []
    color_data = {color: {'x': [], 'y': [], 'sizes': []} for color in colors}
    start_data = {'x': [], 'y': [], 'sizes': []}

    # 1. Go through each point, assign to the right group
    for xi, yi, seq in zip(x, y, sequences):
        if seq in all_starting_seqs:
            # skip coloring this one, it'll be handled as starting point
            continue
        added = False
        for count_map, color in zip(count_maps, colors):
            count = count_map.get(seq, 0)
            if count > threshold:
                color_data[color]['x'].append(xi)
                color_data[color]['y'].append(yi)
                color_data[color]['sizes'].append(count * 2)
                added = True
                break
        if not added:
            gray_x.append(xi)
            gray_y.append(yi)
            gray_sizes.append(3)

    # 2. Handle starting sequences separately (plot as violet)
    for starting_seqs, count_map in zip(starting_seqs_list, count_maps):
        for start_seq in starting_seqs:
            count = count_map.get(start_seq, 0)
            if count > threshold:
                indices = np.where(sequences == start_seq)[0]
                for idx in indices:
                    start_data['x'].append(x[idx])
                    start_data['y'].append(y[idx])
                    start_data['sizes'].append(max((count - 100) * 2, 3))

    # 3. Plotting
    ax.scatter(gray_x, gray_y, c='#D9D9D9', s=gray_sizes,
               alpha=0.2, edgecolors='k', linewidths=0.01)

    for color, lab in zip(colors, labels):
        ax.scatter(color_data[color]['x'], color_data[color]['y'],
                   c=color, s=color_data[color]['sizes'],
                   alpha=0.6, label=f'>{threshold} for {lab}')

    if start_data['x']:
        ax.scatter(start_data['x'], start_data['y'],
                   c='MistyRose', s=start_data['sizes'],
                   alpha=1, edgecolors='black', linewidths=0.1, label='Start')

    ax.set_ylabel("Log stationary distribution")
    ax.set_xlabel("Diffusion axis 1")
    ax.legend()

############################ Dumbbellplot for amino acid frequency changes  ############################

def dumbbellplot(df, cdr=None, order=None, labels=None, naive_color=color_blue, memory_color=color_violet, line_color=2, aa_col=None, naive_col=None, memory_col=None, ax=None):
    """
    Compare naive and memory amino acid frequencies with connected points.
    :param df: pandas.DataFrame: amino acid and naive/memory frequency columns.
    :param cdr: str or None: CDR name used in column inference and plot title.
    :param order: sequence or None: desired amino acid order.
    :param labels: sequence or dict: group labels for occurrence plots, or amino acid label mapping for dumbbell plots.
    :param naive_color: color: color for naive frequencies.
    :param memory_color: color: color for memory frequencies.
    :param line_color: float: width of connecting segments.
    :param aa_col: str or None: amino acid column; inferred when omitted.
    :param naive_col: str or None: naive frequency column; inferred when omitted.
    :param memory_col: str or None: memory frequency column; inferred when omitted.
    :param ax: matplotlib.axes.Axes or None: axes to draw on; None uses current axes.
    :return: matplotlib.axes.Axes: axes containing the plot.
    """

    # --- infer column names if not given explicitly -------------------------
    if aa_col is None:
        if "aa" in df.columns:
            aa_col = "aa"
        elif "aa_code" in df.columns:
            aa_col = "aa_code"
        else:
            raise ValueError("Could not infer amino-acid column (expected 'aa' or 'aa_code').")

    if naive_col is None:
        candidates = []
        if cdr is not None:
            candidates.append(f"naive_{cdr}")
        candidates.append("naive")
        naive_col = next((c for c in candidates if c in df.columns), None)
        if naive_col is None:
            raise ValueError(f"Could not infer naive column from candidates {candidates}.")

    if memory_col is None:
        candidates = []
        if cdr is not None:
            candidates.append(f"memory_{cdr}")
        candidates.append("memory")
        memory_col = next((c for c in candidates if c in df.columns), None)
        if memory_col is None:
            raise ValueError(f"Could not infer memory column from candidates {candidates}.")

    # --- standardize to common column names --------------------------------
    tmp = df[[aa_col, naive_col, memory_col]].rename(
        columns={aa_col: "aa", naive_col: "naive", memory_col: "memory"}
    ).copy()

    # impose order if provided
    if order is not None:
        tmp["aa"] = pd.Categorical(tmp["aa"], categories=order, ordered=True)

    # display labels
    if labels is not None:
        tmp["aa_label"] = tmp["aa"].map(labels)
    else:
        tmp["aa_label"] = tmp["aa"].astype(str)

    # sort rows
    tmp = tmp.sort_values("aa")
    cats = tmp["aa_label"].tolist()

    long = tmp.melt(
        id_vars=["aa", "aa_label"],
        value_vars=["naive", "memory"],
        var_name="state",
        value_name="value",
    )
    long["aa_label"] = pd.Categorical(long["aa_label"], categories=cats, ordered=True)
    long["state"] = pd.Categorical(long["state"], categories=["naive", "memory"], ordered=True)

    if ax is None:
        ax = plt.gca()

    # segments
    for _, r in tmp.iterrows():
        ax.plot(
            [r["naive"], r["memory"]],
            [r["aa_label"], r["aa_label"]],
            color="black",
            lw=float(line_color),
            zorder=1,
        )

    # points
    palette = {"naive": naive_color, "memory": memory_color}
    sn.scatterplot(
        data=long,
        x="value",
        y="aa_label",
        hue="state",
        palette=palette,
        s=80,
        ax=ax,
        alpha=1,
        zorder=2,
    )

    title_cdr = f" — {cdr}" if cdr is not None else ""
    ax.set_title(f"Naive vs Memory{title_cdr}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amino acid")
    ax.margins(x=0.05)
    ax.legend(title="", loc="best")
    sn.despine()
    plt.tight_layout()

    return ax

############################ Add edges to the scatterplot  ############################

def add_edges(connectivity_dict, x_vector1, y_vector2, ax, linewidth = 0.1, alpha = 0.2, color = color_gray, zorder = 0, rasterized = True):

    """
    Add undirected connectivity edges to an existing scatter plot.
    :param connectivity_dict: dict[int, iterable[int]]: index adjacency mapping.
    :param x_vector1: array-like: x coordinates indexed by node.
    :param y_vector2: array-like: y coordinates indexed by node.
    :param ax: matplotlib.axes.Axes or None: axes to draw on; None uses current axes.
    :param linewidth: float: edge width.
    :param alpha: float: edge opacity.
    :param color: color: edge color.
    :param zorder: float: drawing order of edges.
    :param rasterized: bool: rasterize the edge collection in vector output.
    :return: matplotlib.axes.Axes: the supplied axes with a LineCollection added.
    """

    edges = {(min(a,b), max(a,b)) for a, nbrs in connectivity_dict.items() for b in nbrs}
    edges = sorted(edges)

    segments = []
    for u, v in edges:
        segments.append([[x_vector1[u], y_vector2[u]], [x_vector1[v], y_vector2[v]]])

    lc = LineCollection(segments, linewidths=linewidth, rasterized=rasterized, alpha = alpha, zorder = zorder, color = color)
    ax.add_collection(lc)

    return ax


def normal_palette(base_color, n, sigma_h=0, sigma_s=0, sigma_v=0, seed=None):
    """
    Generate colors by adding independent normal noise in HSV space.
    :param base_color: color: Matplotlib-compatible base color.
    :param n: int: number of colors.
    :param sigma_h: float: standard deviation of hue noise.
    :param sigma_s: float: standard deviation of saturation noise.
    :param sigma_v: float: standard deviation of value noise.
    :param seed: int or None: seed for the local random generator.
    :return: list[str]: n hex colors. Hue wraps around; saturation and value are clipped to [0, 1].
    """
    rng = np.random.default_rng(seed)

    ## base hue-saturation-value
    r, g, b = mcolors.to_rgb(base_color)
    H, S, V = mcolors.rgb_to_hsv([r, g, b])

    ### apply jitter
    if sigma_h > 0:
        dh = rng.normal(0.0, sigma_h, size=n)
        ## hue is circular (0 and 1 is the same color), and adding noize can push it below 0 / above 1
        ## => we need value only after decimal point (1.5 -> 0.5)
        H = (H + dh) % 1.0
    else:
        H = [H]*n

    if sigma_s > 0:
        ds = rng.normal(0.0, sigma_s, size=n)
        ## saturation is bounded [0, 1], so we need to clip valued beyond 0 and 1
        S = np.clip(S + ds, 0.0, 1.0)
    else:
        S = [S] * n

    if sigma_v > 0:
        dv = rng.normal(0.0, sigma_v, size=n)
        ## and the same for brightness (value)
        V = np.clip(V + dv, 0.0, 1.0)
    else:
        V = [V] * n

    ## transform back to RGB scale

    cols = []
    for hh, ss, vv in zip(H, S, V):
        rr, gg, bb = mcolors.hsv_to_rgb([hh, ss, vv])
        cols.append(mcolors.to_hex((rr, gg, bb)))
    return cols

############################ Plot trajectory  ############################
def path_edges(seqs_array, sequence_to_index):

    """
    Convert sequence trajectories into undirected index adjacency maps.
    :param seqs_array: sequence of sequences: nucleotide trajectories aligned with times_array.
    :param sequence_to_index: dict[str, int]: sequence to coordinate index.
    :return: list[dict[int, set[int]]]: one adjacency map per trajectory.
    """
    ## construct one undirected adjacency map for each trajectory
    connect_paths = []
    for i in range(len(seqs_array)):
        path = seqs_array[i]
        add_path = {}
        for j in range(len(path)-1):
            item = path[j]
            next_item = path[j+1]
            item_idx = sequence_to_index[item]
            next_item_idx = sequence_to_index[next_item]

            add_path.setdefault(item_idx, set()).add(next_item_idx)
            add_path.setdefault(next_item_idx, set()).add(item_idx)

        connect_paths.append(add_path)

    return connect_paths