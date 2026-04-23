import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass, field
from typing import Callable


# LayerInfo -------------------------------------------------------------------
@dataclass
class LayerInfo:
    """
    Describes one layer's visual representation in the graph.
    
    Attributes:
        n_nodes      : number of nodes (channels / features) in this layer.
        label        : short string shown below the column.
        connectivity : callable(j: int, n_in: int) -> list[int]
                       Given output-node index j and the display-width of the
                       previous layer, return the list of source node indices
                       that j connects from.
        extra_srcs   : list of (src_col_absolute, [src_node_indices]) pairs
                       for skip / residual connections drawn as curved arrows.
    """
    n_nodes : int
    label : str
    connectivity: Callable[[int, int], list[int]] = field(default_factory=lambda: (lambda j, n_in: list(range(n_in))))
    extra_srcs : list[tuple[int, list[int]]] = field(default_factory=list)


# Weight extraction helpers ---------------------------------------------------
def extract_layer_weights(model) -> list[np.ndarray]:
    """Return one (out × in) weight matrix per Linear / Conv1d layer."""
    import torch.nn as nn
    weights = []
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            weights.append(mod.weight.detach().cpu().numpy())
        elif isinstance(mod, nn.Conv1d):
            weights.append(mod.weight.detach().cpu().numpy().mean(axis=2))
    return weights

# Topology from params --------------------------------------------------------
def _build_layers_info(model) -> tuple[list[LayerInfo], str]:
    """
    Build LayerInfo list from ``model.params``.
 
    Connectivity for each layer is derived directly from
    ``params._connections[i]``, the (out × in) bool matrix for transition i.
    ``_connections[i][j, :]`` gives the active source indices for output node j
    at layer i+1.  Skip arrows come from ``params._skip_connections``.
    """
    p = model.params
    layer_sizes  = p.layer_sizes
    n_layers = len(layer_sizes)
    connections  = p._connections       # list of (out × in) bool arrays, len = n_layers-1
    skip_srcs = p._skip_connections  # [j] = [src layer indices]
    layers_info: list[LayerInfo] = []
    
    # for subtitle and label of layers
    kind = "convolutional" if "Conv" in p.name or p.name == "CNN" else "fully connected" if p.name in ("FCNN", "DenseResNet") else "custom"
    
    for L in range(n_layers):
        sz = layer_sizes[L]
 
        # Label ---------------------------------------------------------------
        if L == 0:
            label = f"in\n{sz}n"
        elif L == n_layers - 1:
            label = f"out\n{sz}n"
        else:
            if kind == "convolutional":
                label = f"conv{L}\n{sz}n\nk={model.kernel_size} p={model.padding} s={model.stride}"
            elif kind == "fully connected":
                label = f"dense{L}\n{sz}n"
            else:
                label = f"h{L}\n{sz}n"
        
        # connectivity and skip arrows for this layer -------------------------
        if L == 0 or not connections:
            # input layer — no incoming transitions
            layers_info.append(LayerInfo(n_nodes=sz, label=label))
            continue
        
        # Connectivity from _connections[L-1]: row j gives source indices -----
        conn_matrix = connections[L - 1]   # shape (layer_sizes[L], layer_sizes[L-1])
        
        def _make_conn(mat):
            def _conn(j: int, n_in: int) -> list[int]:
                if j >= mat.shape[0]:
                    return []
                return [int(s) for s in np.where(mat[j, :n_in])[0]]
            return _conn
        
        # Skip arrows from _skip_connections ----------------------------------
        extra: list[tuple[int, list[int]]] = []
        if skip_srcs and L < len(skip_srcs):
            for src_L in sorted(skip_srcs[L]):
                n_src = layer_sizes[src_L]
                extra.append((src_L, list(range(min(n_src, sz)))))
        
        layers_info.append(LayerInfo(n_nodes=sz, label=label, connectivity=_make_conn(conn_matrix), extra_srcs=extra))
    
    # subtitle
    has_skip = p._is_resnet
    convresnet_sub = f" [connect={model.connect}]" if "ConvResNet" in p.name else ""
    subtitle = kind + ("  |  skip connections"+convresnet_sub if has_skip else "")
 
    return layers_info, subtitle

# Edge colour -----------------------------------------------------------------
def _edge_style(weights, w_idx, j, src, max_weight):
    if w_idx is None or w_idx >= len(weights):
        return "#888888", 0.3
    try:
        w     = weights[w_idx][j, src]
        alpha = float(abs(w) / max_weight) * 0.9 + 0.1
        return ("#d62728" if w < 0 else "#1f77b4"), alpha
    except (IndexError, TypeError):
        return "#888888", 0.3
    
# Graph drawing ---------------------------------------------------------------
def draw_graph(ax, model, weights: list | None = None, ellipsize_after: int = 8) -> list:
    """
    Draw the static network graph on *ax*.
    
    Returns edge_artists — list of (Line2D, w_idx, j, src) — for later
    recolouring by update_graph_edges().
    """
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#F5F5F5")
    
    layers_info, subtitle = _build_layers_info(model)
    if weights is None:
        weights = extract_layer_weights(model)
    
    max_weight = max(np.max(np.abs(w)) for w in weights) if weights else 1.0
    n_cols = len(layers_info)
    x_pos = np.linspace(0, (n_cols - 1) * 3, n_cols)
    y_x_ratio = 1.2
    
    def n_shown(ch):
        return min(ch, ellipsize_after)
    
    def y_coords(ch):
        n = n_shown(ch)
        return np.linspace(-(y_x_ratio * n - 1) / 2, (y_x_ratio * n - 1) / 2, n)
    
    # Primary edges
    edge_artists: list = []
    for li in range(1, n_cols):
        info_in, info_out = layers_info[li - 1], layers_info[li]
        ys_in  = y_coords(info_in.n_nodes)
        ys_out = y_coords(info_out.n_nodes)
        n_in   = n_shown(info_in.n_nodes)
        n_out  = n_shown(info_out.n_nodes)
        w_idx  = li - 1 if li - 1 < len(weights) else None
        
        for j in range(n_out):
            for src in info_out.connectivity(j, n_in):
                if 0 <= src < n_in:
                    color, alpha = _edge_style(weights, w_idx, j, src, max_weight)
                    (line,) = ax.plot([x_pos[li - 1], x_pos[li]], [ys_in[src], ys_out[j]], color=color, lw=1.2, alpha=alpha, zorder=1)
                    edge_artists.append((line, w_idx, j, src))
    
    # Skip / residual arrows
    for li, info_out in enumerate(layers_info):
        for src_col, skip_nodes in info_out.extra_srcs:
            ys_skip = y_coords(layers_info[src_col].n_nodes)
            ys_out  = y_coords(info_out.n_nodes)
            n_skip  = n_shown(layers_info[src_col].n_nodes)
            n_out   = n_shown(info_out.n_nodes)
            rad     = 0.2 + 0.1 * (li - src_col)
            for j, src in enumerate(skip_nodes):
                if 0 <= src < n_skip and j < n_out:
                    ax.annotate("", xy=(x_pos[li], ys_out[j]), xytext=(x_pos[src_col], ys_skip[src]), arrowprops=dict(arrowstyle="->", color="#000000", lw=1.5, alpha=0.1, connectionstyle=f"arc3,rad={rad:.2f}"), zorder=1)
    
    # Nodes
    node_r, dot_r = 0.22, 0.06
    for li, info in enumerate(layers_info):
        ys = y_coords(info.n_nodes)
        for i, y in enumerate(ys):
            if info.n_nodes > ellipsize_after and i == len(ys) - 1:
                for dy in (-node_r * 0.9, 0, node_r * 0.9):
                    ax.add_patch(plt.Circle((x_pos[li], y + dy), dot_r, color="#444444", zorder=2, linewidth=0))
            else:
                ax.add_patch(plt.Circle((x_pos[li], y), node_r, facecolor="#5B9BD5", zorder=2, linewidth=0.8, edgecolor="white"))
    
    # Labels
    max_shown = y_x_ratio * max(n_shown(i.n_nodes) for i in layers_info)
    y_bot = -(max_shown / 2) - 0.9
    for li, info in enumerate(layers_info):
        ax.text(x_pos[li], y_bot, info.label, ha="center", va="top",
                fontsize=9, color="#444444", rotation=-15, rotation_mode="anchor")
    
    ax.set_title(f"Network graph  ({subtitle})", pad=4)
    margin = 0.2
    ax.set_xlim(x_pos[0] - margin, x_pos[-1] + margin)
    ax.set_ylim(y_bot - margin, max_shown / 2 + margin)
    
    return edge_artists

def update_graph_edges(edge_artists, weights, max_weight):
    for line, w_idx, j, src in edge_artists:
        color, alpha = _edge_style(weights, w_idx, j, src, max_weight)
        line.set_color(color)
        line.set_alpha(alpha)

# Animation -------------------------------------------------------------------
def make_animation(
    model,
    snapshots:     list,
    loss_history:  list,
    loss2_history: list,
    x_np:          np.ndarray,
    y_np:          np.ndarray,
    pred_color:    str,
    loss_color:    str,
    loss2_color:   str,
    file_type:     str = "",
    savepath:      str = "",
) -> tuple:
    """
    Build a 2×2 animated figure and optionally save it.
    
    Panels:
        top-left     curve fit (animated)
        top-right    network graph (static, edges recoloured each frame)
        bottom-left  primary loss (params.loss_function)
        bottom-right secondary loss (params.loss_function2)
    
    Title and filename come from model.params.
    """
    p = model.params
    epoch_vals = np.arange(1, len(loss_history) + 1)
    all_w = [w for *_, ws in snapshots for w in ws]
    max_weight = max(np.max(np.abs(w)) for w in all_w) if all_w else 1.0
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)
    ax_fit = fig.add_subplot(gs[0, 0])
    ax_graph = fig.add_subplot(gs[0, 1])
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss2 = fig.add_subplot(gs[1, 1])
    
    # Curve fit
    ax_fit.scatter(x_np, y_np, s=8, alpha=0.4, color="#aaaaaa", label="Data", zorder=1)
    ax_fit.set_xlim(x_np.min() / 1.05, x_np.max() * 1.05)
    ax_fit.set_ylim(y_np.min() * 1.2,  y_np.max() * 1.2)
    ax_fit.set_xlabel("x"); ax_fit.set_ylabel("y")
    
    # Network graph
    edge_artists = draw_graph(ax_graph, model, weights=snapshots[0][-1])
    edge_lines = [line for line, *_ in edge_artists]
    
    # Loss panels
    for ax, hist, color, name in ((ax_loss,  loss_history,  loss_color,  type(p.loss_function).__name__+" with noisy y values (used for training)"), (ax_loss2, loss2_history, loss2_color, type(p.loss_function2).__name__+" with noiseless y values")):
        ax.set_xlim(1, epoch_vals[-1])
        ax.set_ylim(max(min(hist) / 1.05, 1e-10), max(hist) * 1.05)
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.set_title(name)
    
    # Dynamic artists
    (line_pred,) = ax_fit.plot([], [], lw=2, color=pred_color, zorder=2, label="Prediction")
    fit_title    = ax_fit.set_title("")
    ax_fit.legend(loc="upper right", fontsize=8)
    
    (line_l,)  = ax_loss.plot([],  [], lw=1.5, color=loss_color)
    dot_l,     = ax_loss.plot([],  [], "o", color=pred_color, ms=6, zorder=3)
    (line_l2,) = ax_loss2.plot([], [], lw=1.5, color=loss2_color)
    dot_l2,    = ax_loss2.plot([], [], "o", color=pred_color, ms=6, zorder=3)
    
    fig.suptitle(p.title, fontsize=13)
    fig.subplots_adjust(top=0.93, hspace=0.35, wspace=0.35)
    
    def update(i):
        epoch, lv, l2v, y_pred, snap_w = snapshots[i]
        mask = epoch_vals <= epoch
        line_pred.set_data(x_np, y_pred)
        fit_title.set_text(f"Epoch {epoch:>4d}")
        line_l.set_data(epoch_vals[mask],  np.array(loss_history)[mask])
        dot_l.set_data([epoch], [lv])
        line_l2.set_data(epoch_vals[mask], np.array(loss2_history)[mask])
        dot_l2.set_data([epoch], [l2v])
        update_graph_edges(edge_artists, snap_w, max_weight)
        return (line_pred, fit_title, line_l, dot_l, line_l2, dot_l2, *edge_lines)
    
    ani = FuncAnimation(fig, update, frames=len(snapshots), interval=80, blit=True, repeat=False)
    
    if file_type:
        os.makedirs(savepath, exist_ok=True)
        name = "".join(p.label.split()) + f".{file_type}"
        path = os.path.join(savepath, name)
        print(f"{name} ... ", end="", flush=True)
        if file_type == "gif":
            ani.save(path, writer="pillow", fps=15)
        else:
            update(len(snapshots) - 1)
            fig.savefig(path)
        print("saved")
    
    return fig, ani