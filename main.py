import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from dataclasses import replace
import argparse

from numpy.ma.core import argmin, argsort

from utils.parameters import ModelParams, apply_seed, FUNC_DICT, LOSS_FUNC_DICT, OPT_DICT
from utils.models import CustomNet
from utils.animation import make_animation
from utils.train import train_model

from optimizer.helpers import compute_edge_importance, apply_mask_to_model
import optimizer.Optimizers as gwo

def test_func(x):
    return 2 * np.exp(-x) * (np.sin(5 * x) + x * np.cos(5 * x))

def min_pruning_mask(importance: np.ndarray, idx: int=0) -> np.ndarray:
    mask_to_prune = np.ones_like(importance)
    i = argsort(importance)
    j = i[idx]
    mask_to_prune[j] = 0
    return mask_to_prune

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
    
    # Configs -----------------------------------------------------------------
    args = parser.parse_args()
    H_N = args.hn
    N_N = args.nn
    FUNC = FUNC_DICT[args.func]
    OPTIMIZER = OPT_DICT[args.opt]
    LOSS1 = LOSS_FUNC_DICT[args.loss1]
    LOSS2 = LOSS_FUNC_DICT[args.loss2]
    LR = args.lr
    TOL = args.tol if args.tol > 0 else None
    EPOCHS = args.epoch
    LOG_EVERY = args.log
    GRAD_CLIP = args.grad_clip if args.grad_clip > 0 else None
    SEED = args.seed if args.seed > 0 else None
    SHUFFLE = args.shuffle
    DEVICE = args.device
    VERBOSE = args.verbose
    NAME = f"CustomNet_{args.name}" if args.name != "" else "CustomNet"
    
    INPUT_N = args.in_n
    NOISE_STD = args.in_std
    SHOW = args.show
    FILE_TYPE = args.file_type
    SAVE = "saves"
    
    PRED_COLOR = "#e05c2e"
    LOSS_COLOR = "#2e7de0"
    LOSS2_COLOR = "#7c3aed"
    
    # Synthetic data (non-uniform Gaussian noise) -----------------------------
    apply_seed(SEED)  # to apply seeds for numpy random functions
    x_np = np.linspace(0, 4, INPUT_N).astype(np.float32)
    y = test_func(x_np)  # the values of the function without noise
    y_np = (y + np.random.normal(size=y.shape, scale=NOISE_STD)).astype(np.float32)  # the values with noise
    
    x_t = torch.tensor(x_np).unsqueeze(1)  # (INPUT_N, 1)
    y_t = torch.tensor(y_np).unsqueeze(1)  # (INPUT_N, 1)
    yr_t = torch.tensor(y).unsqueeze(1)  # (INPUT_N, 1)
    
    os.makedirs(SAVE, exist_ok=True)
    results = []
    
    # ======================================================================= #
    # CustomNet                                                               #
    # ======================================================================= #
    params = ModelParams(
        name="",
        layer_sizes=[1] + [N_N] * H_N + [1],
        activation_functions=[FUNC] * H_N,
        optimizer_function=OPTIMIZER,
        learning_rate=LR,
        max_epoch=EPOCHS,
        print_each=LOG_EVERY,
        gradient_clip=GRAD_CLIP,
        seed=SEED,
        shuffle=SHUFFLE,
        loss_function=LOSS1,
        loss_function2=LOSS2,
        device=DEVICE,
        verbose=VERBOSE)
    
    nodes = []
    layer_sizes = [1] + [N_N] * H_N + [1]  # [1, 3, 3, 3, 1]
    
    for src_layer in range(H_N + 1):
        src_size = layer_sizes[src_layer]
        tgt_layer = src_layer + 1
        for src_node in range(layer_sizes[src_layer]):
            nodes.extend([src_layer, src_node, tgt_layer, tgt_node] for tgt_node in range(layer_sizes[tgt_layer]))
        # if 0 < src_layer < H_N:
        #     for tgt_layer in range(src_layer + 2, H_N + 1):
        #         nodes.extend([src_layer, src_node, tgt_layer, src_node] for src_node in range(layer_sizes[src_layer]))
        
    nums = [layer_sizes[i]*layer_sizes[i+1] for i in range(len(layer_sizes)-1)]
    model = CustomNet(replace(params, name=f"{NAME}"), nodes)
    min_losses = []
    for i in range(1, sum(nums)-H_N):
        snaps, loss, loss2 = train_model(model, x_t, y_t, yr_t)
        min_losses.append(min(loss))
        
        # Early stopping?
        if len(min_losses) > 1 and min_losses[-1] > min_losses[-2]:
            break
        
        imp, edge = compute_edge_importance(model)
        new_model = None
        for ord in range(len(imp)):
            mask = min_pruning_mask(imp, ord)
            new_model = apply_mask_to_model(model, mask, edge)
            if new_model != None:
                continue
        if new_model == None:
            break
        model = new_model
        model.params.name = f"MinWeight_Pruned_i={i}"
    
    # Animate & save
    fig, _ = make_animation(model=model, snapshots=snaps, loss_history=loss, loss2_history=loss2, x_np=x_np,
        y_np=y_np, pred_color=PRED_COLOR, loss_color=LOSS_COLOR, loss2_color=LOSS2_COLOR, file_type=FILE_TYPE,
        savepath=SAVE, )
    
    if SHOW:
        fig.set_size_inches(7, 5)
        plt.show()