import torch.nn as nn
import torch
import numpy as np
from utils.parameters import ModelParams, apply_seed

def _snapshot_weights(model: nn.Module) -> list[np.ndarray]:
    """
    Return a list of weight matrices for every Linear / Conv1d layer in the
    model, in the same order that animation.extract_layer_weights() produces.
    Conv1d weights are averaged over the kernel dimension so every matrix has
    shape (out, in).
    """
    weights = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weights.append(module.weight.detach().cpu().numpy().copy())
        elif isinstance(module, nn.Conv1d):
            w = module.weight.detach().cpu().numpy()
            weights.append(w.mean(axis=2).copy())
    return weights

# Shared training loop --------------------------------------------------------
def train_model(
    model:      nn.Module,
    x:          torch.Tensor,
    y:          torch.Tensor,
    y_r:        torch.Tensor
) -> tuple:
    """
    Train *model* using all hyperparameters stored in ``model.params``.
    
    Reads from ``model.params``
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * ``loss_function``  — primary criterion; tracked as the training loss.
    * ``loss_function2`` — secondary criterion shown in animation panel 4
                           (default ``nn.BCEWithLogitsLoss()``).
    * ``learning_rate``  — Adam learning rate.
    * ``gradient_clip``  — max-norm gradient clipping; ``None`` disables.
    * ``tol``            — early-stop on primary loss; ``None`` disables.
    * ``max_epoch``      — maximum epochs.
    * ``seed``           — re-applied before the training loop so that runs
                           are reproducible even when the model was built with
                           the same seed.
    * ``shuffle``        — if ``True``, samples are randomly permuted at the
                           start of every epoch.
    * ``device``         — tensors are moved here automatically.
    * ``verbose``        — print header + per-epoch progress.
    * ``print_each``     — snapshot + print interval.
    * ``label``          — header string (computed property).
    
    Args:
        model : nn.Module with a ``.params: ModelParams`` attribute.
        x     : input tensor  — moved to ``params.device`` automatically.
        y     : target tensor — moved to ``params.device`` automatically.
    
    Returns:
        snapshots     : list of ``(epoch, loss_val, loss2_val, y_pred_np, weights)``
                        captured every ``print_each`` epochs and at epoch 1.
        loss_history  : primary loss at every epoch.
        loss2_history : secondary loss at every epoch.
    """
    params: ModelParams = model.params
    
    device = params.device
    x = x.to(device)
    y = y.to(device)
    y_r = y_r.to(device)
    apply_seed(params.seed) # Re-seed before training so repeated calls are reproducible
    
    optimizer = params.optimizer_function(model.parameters(), lr=params.learning_rate)
    loss_fn   = params.loss_function
    loss_fn2  = params.loss_function2

    snapshots:  list = []
    loss_history:  list = []
    loss2_history: list = []
    
    n_samples = x.shape[0]
    
    if params.verbose:
        print(f"\n{'-' * 60}")
        print(f"  {params.label}")
        print(model)
        print(f"{'-' * 60}\n")

    for epoch in range(1, params.max_epoch + 1):
        model.train()
        if params.shuffle:
            perm = torch.randperm(n_samples, device=device)
            x_ep  = x[perm]
            y_ep  = y[perm]
            yr_ep = y_r[perm]
        else:
            x_ep  = x
            y_ep  = y
            yr_ep = y_r
        
        optimizer.zero_grad()
        
        pred = model(x_ep)
        loss_val = loss_fn(pred, y_ep)
        loss2_val = loss_fn2(pred, yr_ep)
        loss_val.backward()
        
        if params.gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), params.gradient_clip)
        
        optimizer.step()

        loss_v  = loss_val.item()
        loss2_v = loss2_val.item()
        loss_history.append(loss_v)
        loss2_history.append(loss2_v)
        
        log_this = epoch % params.print_each == 0 or epoch == 1
        
        if log_this:
            with torch.no_grad():
                y_pred = model(x).squeeze().cpu().numpy()
            snapshots.append((epoch, loss_v, loss2_v, y_pred.copy(), _snapshot_weights(model)))
            if params.verbose:
                loss2_name = type(loss_fn2).__name__
                print(f"Epoch {epoch:>6d} | {type(loss_fn).__name__}: {loss_v:.6f} | {loss2_name}: {loss2_v:.4f}")
        
        # Early stopping ------------------------------------------------------
        if params.tol is not None and loss_v < params.tol:
            if params.verbose:
                print(f"Early stop at epoch {epoch} (loss {loss_v:.6e} < tol {params.tol:.6e})")
            if not log_this:
                with torch.no_grad():
                    y_pred = model(x).squeeze().cpu().numpy()
                snapshots.append((epoch, loss_v, loss2_v, y_pred.copy(), _snapshot_weights(model)))
            break
            
    return snapshots, loss_history, loss2_history