"""
main.py
=======
Train a CustomNet for 1-D curve fitting, then run a Binary Grey Wolf Optimizer
to prune its inner edges, minimizing the number of training epochs required to
reach the loss tolerance.

Three-phase pipeline
--------------------
Phase 0  Pre-train the full-density CustomNet → extract weight-importance scores.
Phase 1  GWO searches for the sparsest topology that still converges fastest.
Phase 2  Re-train the winning pruned model from fresh weights and animate.

Helper functions (all GWO-specific, moved here from the old optimizer module)
------------------------------------------------------------------------------
compute_edge_importance   Weight-magnitude importance for inner edges.
apply_mask_to_model       Clone CustomNet with a binary mask applied.
cascade_dead_nodes        Propagate dead-node removal across all layers.
run_gwo_pruning           Full GWO orchestration loop.
plot_gwo_convergence      Save/show the GWO fitness curve.
plot_comparison           Before-vs-after 4-panel comparison figure.
"""

from __future__ import annotations

import copy
import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.parameters import ModelParams, apply_seed, FUNC_DICT, LOSS_FUNC_DICT, OPT_DICT
from utils.models import CustomNet
from utils.animation import make_animation
from utils.train import train_model
from optimizer.opt_parameters import GWOParams, check_gwo_parameters
from optimizer.helpers import compute_edge_importance, apply_mask_to_model
from optimizer.Optimizers import BinaryGWOptimizer


# ===========================================================================
# Target function
# ===========================================================================

def test_func(x: np.ndarray) -> np.ndarray:
    return 2 * np.exp(-x) * (np.sin(5 * x) + x * np.cos(5 * x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script 4 different configurations of neural networks for curve fitting a 2-dimensional data")
    
    # Structure Parameters
    parser.add_argument("--hn", required=False, type=int, default=5,
                        help="Number of hidden layers [type=int, default=5]")
    parser.add_argument("--nn", required=False, type=int, default=7,
                        help="number of nodes at each hidden layer (the number of layers is equal at all hidden layers) [type=int, default=7]")
    parser.add_argument("--func", required=False, type=int, default=17,
                        help="1-26: Which activation function to use, default is Mish [type=int, default=17]"
                             " (to see which number corresponds to what activation function check paramters.py)")
    
    # Training Hyper-parameters
    parser.add_argument("--loss1", required=False, type=int, default=2,
                        help="1-9: Which loss unction to use for the training of models, default is Mean Squared Loss"
                             " [type=int, default=2] (to see which number corresponds to what loss function, check paramters.py)")
    parser.add_argument("--opt", required=False, type=int, default=1,
                        help="1-12: Which optimizer to use for training, default is Adam [type=int, default=1]"
                             " (to see which number corresponds to what optimizer function, check paramters.py)")
    parser.add_argument("--lr", required=False, type=float, default=1e-2,
                        help="Enter learning rate value [type=float, default=1e-2]")
    parser.add_argument("--grad_clip", required=False, type=float, default=100.0,
                        help="Enter the value at which the gradient should be clipped to prevent explosion"
                             " (negative value means not to consider this functionality) [type=float, default=100]")
    parser.add_argument("--tol", required=False, type=float, default=-1,
                        help="Enter the tolerance for the network at which to stop training"
                             " (negative value means not to consider this functionality) [type=float, default=-1]")
    parser.add_argument("--epoch", required=False, type=int, default=5000,
                        help="Number of epochs to run [type=int, default=5000]")
    parser.add_argument("--shuffle", required=False, action="store_true",
                        help="Will shuffle input data of models [action='store_true']")
    parser.add_argument("--device", required=False, type=str, default="cpu",
                        help="What device to use for pytorch [type=str, default='cpu']")
    
    # Outputs and Plots Parameters
    parser.add_argument("--loss2", required=False, type=int, default=6,
                        help="1-9: Which loss unction to use for the second plot (this is not used for training),"
                             " default is Binary Cross Entropy with Logits Loss [type=int, default=6]"
                             " (to see which number corresponds to what loss function, check paramters.py)")
    parser.add_argument("--log", required=False, type=int, default=10,
                        help="the results should should per how many epochs [type=int, default=10]")
    parser.add_argument("--verbose", required=False, action="store_true",
                        help="Whether to show results in console [action='store_true']")
    parser.add_argument("--show", required=False, action="store_true",
                        help="Whether to open figure files after running the code [action='store_true']")
    parser.add_argument("--file_type", required=False, type=str, default="png",
                        help="What should be the file_type of saved figures (gif, png, jpeg) [type=str, default='gif']")
    parser.add_argument("--name", required=False, type=str, default="",
                        help="added string at the end of each file for keeping track at running multiple runs [type=str, default='']")
    
    # Input Data Parameters
    parser.add_argument("--in_n", required=False, type=int, default=200,
                        help="Number of input data [type=int, default=200]")
    parser.add_argument("--in_std", required=False, type=float, default=1e-1,
                        help="Standard deviation for input noise [type=float, default=1e-1]")
    
    # Seed variable influences both training parameters and input data parameters
    parser.add_argument("--seed", required=False, type=int, default=1,
                        help="Seed number for random values (negative value means not to consider this functionality) [type=int, default=1]")

    # GWO hyper-parameters ----------------------------------------------------
    parser.add_argument("--gwo_wolves",  type=int,   default=10,
                        help="GWO: number of wolves [default=10]")
    parser.add_argument("--gwo_iter",    type=int,   default=20,
                        help="GWO: number of iterations [default=20]")
    parser.add_argument("--gwo_prune",   type=float, default=0.4,
                        help="GWO: target prune ratio (0,1) [default=0.4]")
    parser.add_argument("--gwo_norm",    type=int,   default=1,
                        help="GWO: transfer function key 1-7 (see gwo_parameters.py) [default=1=tanh]")
    parser.add_argument("--gwo_a",       type=float, default=1.0,
                        help="GWO: steepness parameter 'a' for transfer function [default=1.0]")
    parser.add_argument("--skip_gwo",    action="store_true",
                        help="Skip GWO pruning — train base model only")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve configs
    # ------------------------------------------------------------------
    H_N       = args.hn
    N_N       = args.nn
    FUNC      = FUNC_DICT[args.func]
    OPTIMIZER = OPT_DICT[args.opt]
    LOSS1     = LOSS_FUNC_DICT[args.loss1]
    LOSS2     = LOSS_FUNC_DICT[args.loss2]
    LR        = args.lr
    TOL       = args.tol if args.tol > 0 else None
    EPOCHS    = args.epoch
    LOG_EVERY = args.log
    GRAD_CLIP = args.grad_clip if args.grad_clip > 0 else None
    SEED      = args.seed if args.seed >= 0 else None
    SHUFFLE   = args.shuffle
    DEVICE    = args.device
    VERBOSE   = args.verbose
    NAME      = f"_{args.name}" if args.name else ""

    INPUT_N   = args.in_n
    NOISE_STD = args.in_std
    SHOW      = args.show
    FILE_TYPE = args.file_type
    SAVE      = "saves"

    PRED_COLOR  = "#e05c2e"
    LOSS_COLOR  = "#2e7de0"
    LOSS2_COLOR = "#7c3aed"

    os.makedirs(SAVE, exist_ok=True)

    # ------------------------------------------------------------------
    # Synthetic data
    # ------------------------------------------------------------------
    apply_seed(SEED)
    x_np      = np.linspace(0, 4, INPUT_N).astype(np.float32)
    y_true_np = test_func(x_np)
    y_np      = (y_true_np + np.random.normal(size=y_true_np.shape, scale=NOISE_STD)).astype(np.float32)

    x_t  = torch.tensor(x_np).unsqueeze(1)
    y_t  = torch.tensor(y_np).unsqueeze(1)
    yr_t = torch.tensor(y_true_np).unsqueeze(1)

    # ------------------------------------------------------------------
    # Build CustomNet
    # ------------------------------------------------------------------
    layer_sizes = [1] + [N_N] * H_N + [1]

    model_params = ModelParams(
        name                 = f"CustomNet{NAME}",
        layer_sizes          = layer_sizes,
        activation_functions = [FUNC] * H_N,
        optimizer_function   = OPTIMIZER,
        learning_rate        = LR,
        max_epoch            = EPOCHS,
        print_each           = LOG_EVERY,
        gradient_clip        = GRAD_CLIP,
        seed                 = SEED,
        shuffle              = SHUFFLE,
        loss_function        = LOSS1,
        loss_function2       = LOSS2,
        device               = DEVICE,
        verbose              = VERBOSE,
        tol                  = TOL,
    )
    
    nodes = []
    for src_layer in range(H_N + 1):
        src_size = layer_sizes[src_layer]
        tgt_layer = src_layer + 1
        for src_node in range(layer_sizes[src_layer]):
            nodes.extend([src_layer, src_node, tgt_layer, tgt_node] for tgt_node in range(layer_sizes[tgt_layer]))
            # if 0 < src_layer < H_N:
            #     for tgt_layer in range(src_layer + 2, H_N + 1):
            #         nodes.extend([src_layer, src_node, tgt_layer, src_node] for src_node in range(layer_sizes[src_layer]))
    
    base_model = CustomNet(model_params, nodes)

    # ------------------------------------------------------------------
    # Phase 1 — train the base model
    # (also serves as GWO pre-training for importance extraction)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Phase 1 — Training base CustomNet")
    print("=" * 60)

    base_snaps, base_loss_hist, base_loss2_hist = train_model(base_model, x_t, y_t, yr_t)

    print("\nSaving base model animation ...")
    make_animation(
        model         = base_model,
        snapshots     = base_snaps,
        loss_history  = base_loss_hist,
        loss2_history = base_loss2_hist,
        x_np          = x_np,
        y_np          = y_np,
        pred_color    = PRED_COLOR,
        loss_color    = LOSS_COLOR,
        loss2_color   = LOSS2_COLOR,
        file_type     = FILE_TYPE,
        savepath      = SAVE,
    )

    # ------------------------------------------------------------------
    # Phase 2 — GWO pruning
    # ------------------------------------------------------------------
    if not args.skip_gwo:
        gwo_params = GWOParams(
            n_wolves       = args.gwo_wolves,
            max_iter       = args.gwo_iter,
            prune_ratio    = args.gwo_prune,
            normalize_func = args.gwo_norm,
            normalize_a    = args.gwo_a,
            seed           = SEED,
            verbose        = True,
        )
        check_gwo_parameters(gwo_params)

        print("\n" + "=" * 60)
        print("  Phase 2 — GWO Edge Pruning")
        print(f"  Transfer function : {gwo_params.normalize_name}")
        print("=" * 60)
        
        # 1. Importance extraction ------------------------------------------------
        importance: np.ndarray
        edge_index: list[tuple[int, int, int, int]]
        (importance, edge_index) = compute_edge_importance(base_model)
        n_inner = len(edge_index)
        
        if n_inner == 0:
            if gwo_params.verbose:
                raise ValueError("  No inner edges found — returning base model unchanged.")
        
        if gwo_params.verbose:
            print(f"\n  Inner edges eligible for pruning : {n_inner}")
            print(f"  Transfer function : {gwo_params.normalize_name}  (a={gwo_params.normalize_a})")
            print(f"  Wolves={gwo_params.n_wolves}, iterations={gwo_params.max_iter}, "
                  f"prune_ratio={gwo_params.prune_ratio}\n")
        
        # 2. Initialise GWO -------------------------------------------------------
        gwo = BinaryGWOptimizer(gwo_params=gwo_params, n_edges=n_inner, importance=importance)
        best_mask: np.ndarray = np.ones(n_inner, dtype=np.int8)
        best_score: float = float("inf")
        score_history: list[float] = []
        best_result = (base_model, base_snaps, base_loss_hist, base_loss2_hist)
        
        convergence_curve: list[float] = []
        
        # 3–4. Main GWO loop ------------------------------------------------------
        for it in range(gwo_params.max_iter):
            masks = gwo.step(it)  # (n_wolves, n_inner)
            scores = np.full(gwo_params.n_wolves, float("inf"), dtype=float)
            
            for w_idx in range(gwo_params.n_wolves):
                mask = masks[w_idx]
                candidate = apply_mask_to_model(base_model, mask, edge_index)
                if candidate is None:
                    continue
                
                # Silence verbose during fitness evaluation
                orig_verbose = candidate.params.verbose
                candidate.params.verbose = False
                
                snaps, lh, lh2 = train_model(candidate, x_t, y_t, yr_t)
                
                candidate.params.verbose = orig_verbose
                scores[w_idx] = min(lh)
                
                if scores[w_idx] < best_score:
                    best_score = scores[w_idx]
                    best_mask = mask.copy()
                    best_result = (candidate, snaps, lh, lh2)
            
            # Update leaders
            gwo.update_leaders(scores)
            convergence_curve.append(gwo.alpha.score)
            score_history.append(gwo.alpha.score)
            
            if gwo_params.verbose:
                kept = int(best_mask.sum())
                print(f"  [GWO iter {it + 1:>3d}/{gwo_params.max_iter}] "
                      f"alpha={gwo.alpha.score:.4f}  "
                      f"best_fitness={best_score:.4f}  "
                      f"kept_inner={kept}/{n_inner}")
        
        if gwo_params.verbose:
            n_pruned = n_inner - int(best_mask.sum())
            print(f"\n  GWO finished — pruned {n_pruned}/{n_inner} inner edges.")
            print(f"  Best fitness = {best_score:.4f}\n")
        
        (pruned_model, pruned_snaps, pruned_loss_hist, pruned_loss2_hist) = best_result
        pruned_model.params.name = f"GWO_Pruned{NAME}"
        print("\nSaving pruned model animation ...")
        make_animation(
            model         = pruned_model,
            snapshots     = pruned_snaps,
            loss_history  = pruned_loss_hist,
            loss2_history = pruned_loss2_hist,
            x_np          = x_np,
            y_np          = y_np,
            pred_color    = PRED_COLOR,
            loss_color    = LOSS_COLOR,
            loss2_color   = LOSS2_COLOR,
            file_type     = FILE_TYPE,
            savepath      = SAVE,
        )

        # # GWO convergence plot
        # importance, edge_index = compute_edge_importance(base_model)
        # plot_gwo_convergence(
        #     score_history = score_history,
        #     n_inner_edges = len(edge_index),
        #     best_score    = best_score,
        #     savepath      = SAVE,
        #     name          = NAME,
        #     file_type     = FILE_TYPE,
        #     show          = SHOW,
        # )

        # # Before-vs-after comparison
        # plot_comparison(
        #     x_np             = x_np,
        #     y_np             = y_np,
        #     y_true           = y_true_np,
        #     x_t              = x_t,
        #     base_model       = base_model,
        #     pruned_model     = pruned_model,
        #     base_loss_hist   = base_loss_hist,
        #     pruned_loss_hist = pruned_loss_hist,
        #     savepath         = SAVE,
        #     name             = NAME,
        #     file_type        = FILE_TYPE,
        #     show             = SHOW,
        # )
    
    if SHOW:
        plt.show()
