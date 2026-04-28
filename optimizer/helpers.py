import operator
import os
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.ma.core import argmin
from sympy import true

from utils.train import train_model
from utils.models import CustomNet
from optimizer.opt_parameters import GWOParams
from optimizer.Optimizers import BinaryGWOptimizer

# ===========================================================================
# GWO helper — weight-importance extraction
# ===========================================================================
def compute_edge_importance(model: CustomNet, ) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Compute a normalized importance score for every *inner* adjacent edge
    (edges that are **not** in the first transition src_layer=0 or the last
    transition tgt_layer=n_layers-1).

    Importance = normalized absolute weight value.
    Higher importance → edge more likely to be kept.

    Parameters
    ----------
    model : a trained ``CustomNet`` instance.

    Returns
    -------
    importance  : float64 array of shape ``(n_inner_edges,)`` in [0, 1].
    edge_index  : list of ``(src_layer, src_node, tgt_layer, tgt_node)``
                  tuples in the same order as *importance*.
    """
    n_layers = model.n_layers
    connections = model.params._connections  # list of (out×in) bool arrays
    linears = list(model._linears)  # one Linear per transition
    
    edge_index: list[tuple[int, int, int, int]] = []
    raw_weights: list[float] = []
    
    for trans_idx, (conn, lin) in enumerate(zip(connections, linears)):
        src_layer = trans_idx
        tgt_layer = trans_idx + 1
        
        # Exclude first and last transitions from the search space
        if src_layer == 0 or tgt_layer == n_layers - 1:
            continue
        
        weight = lin.weight.detach().cpu().numpy()  # (tgt_size × src_size)
        tgt_size, src_size = conn.shape
        
        for tgt_node in range(tgt_size):
            for src_node in range(src_size):
                if conn[tgt_node, src_node]:
                    edge_index.append((src_layer, src_node, tgt_layer, tgt_node))
                    raw_weights.append(abs(float(weight[tgt_node, src_node])))
    
    if not raw_weights:
        return np.array([], dtype=np.float64), []
    
    raw = np.array(raw_weights, dtype=np.float64)
    w_min, w_max = raw.min(), raw.max()
    importance = (raw - w_min) / (w_max - w_min) if w_max > w_min else np.full_like(raw, 0.5)
    return importance, edge_index


# ===========================================================================
# GWO helper — dead-node cascade pruning
# ===========================================================================
def cascade_dead_nodes(surviving_nodes: list[tuple[int, int, int, int]],
        layer_sizes: list[int], n_layers: int, ) -> list[tuple[int, int, int, int]]:
    """
        Remove all edges connected to *dead nodes* and propagate until stable.

        A node is *dead* if it has no incoming edges (and is not an input-layer
        node) **or** no outgoing edges (and is not an output-layer node).

        Rules
        -----
        * Input-layer nodes (layer 0) are *never* declared dead regardless of
          their outgoing edges — the input dimension must stay fixed.
        * Output-layer nodes (layer n_layers-1) are *never* declared dead
          regardless of their incoming edges — the output dimension must stay fixed.
        * When a hidden node becomes dead, **all** its edges are removed — including
          edges that touch the first or last transition (i.e. edges from input layer
          to this hidden node, or from this hidden node to output layer).

        The function iterates until no further nodes are pruned.

        Parameters
        ----------
        surviving_nodes : list of ``(src_layer, src_node, tgt_layer, tgt_node)``
                          that remain after the GWO binary mask is applied.
        layer_sizes     : ``model.layer_sizes``.
        n_layers        : ``model.n_layers``.

        Returns
        -------
        Cleaned node list (subset of *surviving_nodes*).
        """
    nodes = list(surviving_nodes)
    while True:
        has_incoming: dict[tuple[int, int], bool] = {}
        has_outgoing: dict[tuple[int, int], bool] = {}
        for src_l, src_n, tgt_l, tgt_n in nodes:
            has_outgoing[(src_l, src_n)] = True
            has_incoming[(tgt_l, tgt_n)] = True
        dead: list[tuple[int, int]] = []
        for layer in range(1, n_layers-1): # Input nodes and output nodes: never dead (fixed input dimension)
            for node in range(layer_sizes[layer]):
                key = (layer, node)
                # Hidden node: dead if it has no incoming OR no outgoing edges
                # but if no outgoing and no incoming then the node has been fully pruned and is dead already
                if operator.xor(not has_incoming.get(key, False), not has_outgoing.get(key, False)):
                    dead.append(key)
        if len(dead) == 0:
            break  # Stable — no more dead nodes
        dead_set = set(dead)
        nodes = [(sl, sn, tl, tn) for sl, sn, tl, tn in nodes if (sl, sn) not in dead_set and (tl, tn) not in dead_set]
    return nodes

# def cascade_dead_nodes(surviving_nodes: list[tuple[int, int, int, int]],
#         layer_sizes: list[int], n_layers: int, ) -> list[tuple[int, int, int, int]]:
#
#     # Build incoming / outgoing adjacency sets for every (layer, node)
#     has_incoming: dict[tuple[int, int], bool] = {}
#     has_outgoing: dict[tuple[int, int], bool] = {}
#
#     for src_l, src_n, tgt_l, tgt_n in surviving_nodes:
#         has_outgoing[(src_l, src_n)] = True
#         has_incoming[(tgt_l, tgt_n)] = True
#
#     dead: list[tuple[int, int]] = []
#     nodes: list[tuple[int, int, int, int]] = []
#
#     for layer in range(n_layers):
#         for node in range(layer_sizes[layer]):
#             key = (layer, node)
#             if layer == 0 or layer == n_layers - 1:
#                 # Input nodes and output nodes: never dead (fixed input dimension)
#                 continue
#             # Hidden node: dead if it has no incoming OR no outgoing edges
#             # but if no outgoing and no incoming then the node has been fully pruned and is dead already
#             if operator.xor(not has_incoming.get(key, False), not has_outgoing.get(key, False)):
#                 dead.append(key)
#
#     if len(dead) == 0:
#         return surviving_nodes  # Stable — no more dead nodes
#
#     # Remove every edge that touches the dead node (src or tgt)
#     for (sl, sn, tl, tn) in surviving_nodes:
#         if ((sl, sn) not in dead) and ((tl, tn) not in dead):
#             nodes.append((sl, sn, tl, tn))
#     return cascade_dead_nodes(nodes, layer_sizes, n_layers)


# ===========================================================================
# GWO helper — apply binary mask → new CustomNet
# ===========================================================================
def apply_mask_to_model(base_model: CustomNet, mask: np.ndarray,
        edge_index: list[tuple[int, int, int, int]], ) -> CustomNet | None:
    """
    Clone *base_model* and prune every inner edge where ``mask == 0``,
    then cascade dead-node removal.

    The first/last-layer edges are kept as-is in the initial surviving set;
    the dead-node cascade may remove some of them if hidden nodes they
    connect to become isolated, **without** shrinking the input/output sizes.

    Parameters
    ----------
    base_model  : the original ``CustomNet`` (untouched).
    mask        : int8 array of shape ``(n_inner_edges,)``; 0 → prune.
    edge_index  : inner edges in the same order as *mask*.

    Returns
    -------
    A new ``CustomNet`` with the pruned topology, or ``None`` if the resulting
    graph is degenerate (output layer unreachable from input layer).
    """
    p = base_model.params
    n_layers = base_model.n_layers
    layer_sizes = base_model.layer_sizes
    
    # Build set of inner edges to prune according to the mask
    pruned_inner: set[tuple] = {edge for edge, keep in zip(edge_index, mask) if keep == 0}
    
    # Start with ALL currently active edges (including first/last transitions)
    all_active: list[tuple[int, int, int, int]] = []
    for trans_idx, conn in enumerate(p._connections):
        src_layer = trans_idx
        tgt_layer = trans_idx + 1
        tgt_size, src_size = conn.shape
        for tgt_node in range(tgt_size):
            for src_node in range(src_size):
                if conn[tgt_node, src_node]:
                    all_active.append((src_layer, src_node, tgt_layer, tgt_node))
    
    # Remove GWO-pruned inner edges
    surviving = [e for e in all_active if e not in pruned_inner]
    
    # Cascade dead-node removal (may also drop some first/last-layer edges)
    surviving = cascade_dead_nodes(surviving, layer_sizes, n_layers)

    # Sanity check: can we still reach the output from the input?
    reachable = [False] * n_layers
    reachable[0] = True
    for sl, sn, tl, tn in surviving:
        if reachable[sl]:
            reachable[tl] = True
    if not reachable[n_layers - 1]:
        return None  # Degenerate — skip this wolf
    
    # Clone params and construct the pruned CustomNet
    new_params = copy.deepcopy(p)
    new_params._connections = []
    new_params._pruned = set()
    new_params._skip_connections = []
    
    return CustomNet(new_params, surviving)

# # ===========================================================================
# # Plotting helpers
# # ===========================================================================
# def plot_gwo_convergence(score_history: list[
#     float], n_inner_edges: int, best_score: float, savepath: str, name: str, file_type: str, show: bool, ) -> None:
#     """Save / show the GWO alpha-fitness convergence curve."""
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.plot(range(1, len(score_history) + 1), score_history, color="#e05c2e", lw=2, marker="o", ms=5)
#     ax.set_xlabel("GWO Iteration")
#     ax.set_ylabel("Alpha fitness (epochs to converge + sparsity bonus)")
#     ax.set_title(f"GWO Pruning Convergence\n"
#                  f"Inner edges eligible: {n_inner_edges}  |  Best fitness: {best_score:.4f}")
#     ax.grid(True, alpha=0.3)
#     fig.tight_layout()
#
#     if file_type:
#         os.makedirs(savepath, exist_ok=True)
#         fname = f"GWO_convergence{name}.{file_type}"
#         fig.savefig(os.path.join(savepath, fname))
#         print(f"Saved: {fname}")
#
#     if show:
#         plt.show()
#     plt.close(fig)
#
#
# def plot_comparison(x_np: np.ndarray, y_np: np.ndarray, y_true: np.ndarray, x_t: torch.Tensor, base_model: CustomNet, pruned_model: CustomNet, base_loss_hist:
# list[float], pruned_loss_hist: list[float], savepath: str, name: str, file_type: str, show: bool, ) -> None:
#     """
#     4-panel comparison figure:
#       top-left   : base model curve fit
#       top-right  : pruned model curve fit
#       bottom-left: overlaid training loss curves
#       bottom-right: epochs-to-converge bar chart
#     """
#     tol = base_model.params.tol
#
#     base_model.eval()
#     pruned_model.eval()
#
#     with torch.no_grad():
#         y_base = base_model(x_t).squeeze().cpu().numpy()
#         y_pruned = pruned_model(x_t).squeeze().cpu().numpy()
#
#     def epochs_to_tol(hist: list[float]) -> int:
#         if tol is None:
#             return len(hist)
#         for i, v in enumerate(hist):
#             if v < tol:
#                 return i + 1
#         return len(hist)
#
#     base_conv = epochs_to_tol(base_loss_hist)
#     pruned_conv = epochs_to_tol(pruned_loss_hist)
#
#     def count_inner_edges(m: CustomNet) -> int:
#         n = m.n_layers
#         total = 0
#         for t, conn in enumerate(m.params._connections):
#             if t == 0 or t == n - 2:
#                 continue
#             total += int(conn.sum())
#         return total
#
#     base_inner = count_inner_edges(base_model)
#     pruned_inner = count_inner_edges(pruned_model)
#
#     fig = plt.figure(figsize=(14, 9))
#     gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)
#
#     # top-left: base model fit ------------------------------------------------
#     ax0 = fig.add_subplot(gs[0, 0])
#     ax0.scatter(x_np, y_np, s=8, alpha=0.35, color="#aaaaaa", label="Noisy data")
#     ax0.plot(x_np, y_true, lw=1.5, color="#555555", ls="--", label="True function")
#     ax0.plot(x_np, y_base, lw=2, color="#2e7de0", label="Base model")
#     ax0.set_title(f"Base model — curve fit  (inner edges: {base_inner})")
#     ax0.set_xlabel("x");
#     ax0.set_ylabel("y")
#     ax0.legend(fontsize=8)
#
#     # top-right: pruned model fit ---------------------------------------------
#     ax1 = fig.add_subplot(gs[0, 1])
#     ax1.scatter(x_np, y_np, s=8, alpha=0.35, color="#aaaaaa", label="Noisy data")
#     ax1.plot(x_np, y_true, lw=1.5, color="#555555", ls="--", label="True function")
#     ax1.plot(x_np, y_pruned, lw=2, color="#e05c2e", label="GWO-pruned model")
#     ax1.set_title(f"GWO-pruned model — curve fit  (inner edges: {pruned_inner})")
#     ax1.set_xlabel("x");
#     ax1.set_ylabel("y")
#     ax1.legend(fontsize=8)
#
#     # bottom-left: loss curves ------------------------------------------------
#     ax2 = fig.add_subplot(gs[1, 0])
#     ax2.plot(range(1, len(base_loss_hist) + 1), base_loss_hist, lw=1.5, color="#2e7de0", label="Base")
#     ax2.plot(range(1, len(pruned_loss_hist) + 1), pruned_loss_hist, lw=1.5, color="#e05c2e", label="GWO-pruned")
#     if tol is not None:
#         ax2.axhline(tol, color="#7c3aed", ls=":", lw=1.2, label=f"tol={tol:.0e}")
#     ax2.set_yscale("log")
#     ax2.set_xlabel("Epoch");
#     ax2.set_ylabel("Loss (log scale)")
#     ax2.set_title("Training loss — base vs. pruned")
#     ax2.legend(fontsize=8)
#
#     # bottom-right: epochs-to-converge bar ------------------------------------
#     ax3 = fig.add_subplot(gs[1, 1])
#     labels = ["Base model", "GWO-pruned"]
#     values = [base_conv, pruned_conv]
#     colors = ["#2e7de0", "#e05c2e"]
#     bars = ax3.bar(labels, values, color=colors, width=0.4, edgecolor="white")
#     top = max(values) if max(values) > 0 else 1
#     for bar, val in zip(bars, values):
#         ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.01, str(val), ha="center", va="bottom",
#                  fontsize=11, fontweight="bold")
#     ax3.set_ylabel("Epochs to reach tol")
#     ax3.set_title("Convergence speed")
#     ax3.set_ylim(0, top * 1.2)
#
#     removed = base_inner - pruned_inner
#     fig.suptitle(f"GWO Edge Pruning — inner edges: {base_inner} → {pruned_inner}  "
#                  f"({removed} removed, {removed / max(base_inner, 1) * 100:.1f}%)", fontsize=12, )
#
#     if file_type:
#         os.makedirs(savepath, exist_ok=True)
#         fname = f"GWO_comparison{name}.{file_type}"
#         fig.savefig(os.path.join(savepath, fname))
#         print(f"Saved: {fname}")
#
#     if show:
#         plt.show()
#     plt.close(fig)