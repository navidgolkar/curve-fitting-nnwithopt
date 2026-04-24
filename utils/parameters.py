from __future__ import annotations

import random
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field

FUNC_DICT = {
        1: nn.ELU(),
        2: nn.Hardshrink(),
        3: nn.Hardsigmoid(),
        4: nn.Hardtanh(),
        5: nn.Hardswish(),
        6: nn.LeakyReLU(),
        7: nn.LogSigmoid(),
        8: nn.PReLU(),
        9: nn.ReLU(),
        10: nn.ReLU6(),
        11: nn.RReLU(),
        12: nn.SELU(),
        13: nn.CELU(),
        14: nn.GELU(),
        15: nn.Sigmoid(),
        16: nn.SiLU(),
        17: nn.Mish(),
        18: nn.Softplus(),
        19: nn.Softshrink(),
        20: nn.Softsign(),
        21: nn.Tanh(),
        22: nn.Tanhshrink(),
        23: nn.Threshold(threshold = 0, value = -0.1), # with threshold=0 and value=0 would be the same as ReLU 
        24: nn.GLU(),
        25: nn.Softmin(),
        26: nn.Softmax(),
        27: nn.LogSoftmax(),
        }

LOSS_FUNC_DICT = {
    1: nn.L1Loss(),
    2: nn.MSELoss(),
    3: nn.HuberLoss(),
    4: nn.SmoothL1Loss(),
    5: nn.PoissonNLLLoss(),
    6: nn.BCEWithLogitsLoss(),
    7: nn.HingeEmbeddingLoss(),
    8: nn.SoftMarginLoss(),
    9: nn.MultiLabelSoftMarginLoss(),
    # 10: nn.CrossEntropyLoss(),
    # 11: nn.CTCLoss(),
    # 12: nn.NLLLoss(),
    # 13: nn.GaussianNLLLoss(),
    # 14: nn.KLDivLoss(),
    # 15: nn.BCELoss(),
    # 16: nn.MarginRankingLoss(),
    # 17: nn.HingeEmbeddingLoss(),
    # 18: nn.MultiLabelMarginLoss(),
    # 19: nn.MultiLabelSoftMarginLoss(),
    # 20: nn.CosineEmbeddingLoss(),
    # 21: nn.MultiMarginLoss(),
    # 22: nn.TripletMarginLoss(),
    # 23: nn.TripletMarginWithDistanceLoss(),
    }

OPT_DICT = {
    
    1: lambda params, lr: torch.optim.Adam(params, lr=lr),
    2: lambda params, lr: torch.optim.AdamW(params, lr=lr),
    3: lambda params, lr: torch.optim.Adamax(params, lr=lr),
    4: lambda params, lr: torch.optim.NAdam(params, lr=lr),
    5: lambda params, lr: torch.optim.RMSprop(params, lr=lr),
    6: lambda params, lr: torch.optim.RAdam(params, lr=lr),
    7: lambda params, lr: torch.optim.Adafactor(params, lr=lr),
    8: lambda params, lr: torch.optim.Rprop(params, lr=lr),
    9: lambda params, lr: torch.optim.Adagrad(params, lr=lr),
    10: lambda params, lr: torch.optim.Adadelta(params, lr=lr),
    11: lambda params, lr: torch.optim.ASGD(params, lr=lr),
    12: lambda params, lr: torch.optim.SGD(params, lr=lr),
    # 13: lambda params, lr: torch.optim.LBFGS(params, lr=lr),
    # 14: lambda params, lr: torch.optim.SparseAdam(params, lr=lr),
    # 15: lambda params, lr: torch.optim.Muon(params, lr=lr),
}

@dataclass
class ModelParams:
    """
    Shared configuration for every model in models.py.

    Attributes
    ----------
    Architecture / identity
    ~~~~~~~~~~~~~~~~~~~~~~~
    name : str
        Human-readable model class name, e.g. ``"FCNN"``, ``"ConvResNet"``.
    layer_sizes : list[int]
        Number of nodes / filters in each layer.  ``layer_sizes[0]`` is the
        input width; ``layer_sizes[-1]`` is the output width.

    Graph state  *(populated by the model constructor — do not set manually)*
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _connections : list[np.ndarray]
        ``_connections[i]`` is a boolean ``(layer_sizes[i+1] × layer_sizes[i])``
        matrix.  Entry ``[out, in] = True`` iff that edge is active (not pruned).
        Length = ``len(layer_sizes) - 1`` (one matrix per adjacent transition,
        including the output head).  Populated by the model constructor.
        For CustomNet, updated in-place by ``add_pruned()``.
    _pruned : set[tuple[int, int, int, int]]
        ``(src_layer, src_node, tgt_layer, tgt_node)`` edges whose weights are
        forced to 0.  Starts empty; updated via ``model.add_pruned()``.
    _skip_connections : list[list[int]]
        ``_skip_connections[j]`` = source layer indices whose output is added
        residual-style to layer ``j``'s input.  Mirrors ``_skip_sources``.
    _is_resnet : bool
        ``True`` when the model has residual / skip connections.

    Training
    ~~~~~~~~
    learning_rate : float
        Adam optimiser learning rate.  Must be > 0.
    gradient_clip : float | None
        Max-norm for ``torch.nn.utils.clip_grad_norm_``.  ``None`` disables.
        When set, must be > 0.
    loss_function : nn.Module
        Primary loss criterion shown in animation panel 3, e.g.
        ``nn.MSELoss()``.
    loss_function2 : nn.Module
        Secondary loss criterion shown in animation panel 4.  Defaults to
        ``nn.BCEWithLogitsLoss()``.
    tol : float | None
        Early-stopping threshold on the *primary* loss.  Training stops when
        ``loss < tol``.  ``None`` disables.  When set, must be > 0.
    max_epoch : int
        Maximum number of training epochs.  Must be ≥ 1.
    activation_functions : list[nn.Module]
        One activation per layer transition.  Length must equal
        ``len(layer_sizes) - 1``.  Use ``nn.Identity()`` for a linear (no-op).
    seed : int | None
        Global random seed applied to ``random``, ``numpy``, ``torch``, and
        ``torch.cuda`` before weight initialisation and before each training
        run.  ``None`` disables deterministic seeding.  When set must be a
        non-negative integer.
    shuffle : bool
        Shuffle training samples at the start of every epoch.  Default
        ``True``.

    Runtime / logging
    ~~~~~~~~~~~~~~~~~
    device : str
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, ``"mps"``.
    verbose : bool
        Print training progress to stdout.
    print_each : int
        Print and snapshot every *print_each* epochs.  Must be ≥ 1.

    Computed
    --------
    label : str  *(read-only property)*
     File-safe identifier: ``"{name}_{n_layers}x{max_width}_{seed}"``.
     title : str  *(read-only property)*
         Human-readable title for figure suptitle.
     """

    # Architecture / identity -------------------------------------------------
    name:        str
    layer_sizes: list[int]

    # Graph state (populated by model constructors) ---------------------------
    _connections:      list[np.ndarray]                  = field(default_factory=list)
    _pruned:           set[tuple[int, int, int, int]]    = field(default_factory=set)
    _skip_connections: list[list[int]]                   = field(default_factory=list)
    _is_resnet:        bool                              = False
    
    # Training ----------------------------------------------------------------
    learning_rate:        float                 = 1e-3
    gradient_clip:        float | None          = None
    optimizer_function:   torch.optim.Optimizer = field(default_factory=torch.optim.Adam)
    loss_function:        nn.Module             = field(default_factory=nn.MSELoss)
    loss_function2:       nn.Module             = field(default_factory=nn.BCEWithLogitsLoss)
    tol:                  float | None          = None
    max_epoch:            int                   = 1000
    activation_functions: list[nn.Module]       = field(default_factory=list)
    seed:                 int | None            = None
    shuffle:              bool                  = True

    # Runtime / logging -------------------------------------------------------
    device:      str  = "cpu"
    verbose:     bool = True
    print_each:  int  = 100

    # Computed ----------------------------------------------------------------
    @property
    def label(self) -> str:
        width = max(self.layer_sizes) if self.layer_sizes else 0
        return f"{self.name}_{len(self.layer_sizes)}x{width}_{self.seed}"
    
    @property
    def title(self) -> str:
        width = max(self.layer_sizes) if self.layer_sizes else 0
        return f"{self.name} | Seed = {self.seed} | {len(self.layer_sizes)-2} Hidden Layers x {width} Nodes"


# -----------------------------------------------------------------------------
# Seeding helper
# -----------------------------------------------------------------------------

def apply_seed(seed: int | None) -> None:
    """
    Seed all sources of randomness used across the project.

    Covers ``random``, ``numpy``, ``torch`` (CPU and CUDA), and sets
    ``torch.backends.cudnn.deterministic = True`` /
    ``torch.backends.cudnn.benchmark = False`` for full GPU reproducibility.
    Does nothing when *seed* is ``None``.

    Args:
        seed: Non-negative integer seed, or ``None`` to skip seeding.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# -----------------------------------------------------------------------------
# Parameter validation
# -----------------------------------------------------------------------------

def check_parameters(p: ModelParams) -> None:
    """
    Validate a ``ModelParams`` instance and raise ``ValueError`` listing *all*
    problems found (not just the first).

    Checks performed
    ~~~~~~~~~~~~~~~~
    *Architecture*
        - ``name`` is a non-empty string.
        - ``layer_sizes`` is a list of positive integers.

    *Training*
        - ``learning_rate`` is a positive number.
        - ``gradient_clip`` is ``None`` or a positive number.
        - ``loss_function`` is an ``nn.Module`` instance.
        - ``loss_function2`` is an ``nn.Module`` instance.
        - ``tol`` is ``None`` or a positive number.
        - ``max_epoch`` is an integer ≥ 1.
        - ``activation_functions`` is a list of exactly number of layers - 2 (all layers except input and output)
          ``nn.Module`` instances.
        - ``seed`` is ``None`` or a non-negative integer.
        - ``shuffle`` is a boolean.

    *Runtime*
        - ``device`` is a non-empty string.
        - ``verbose`` is a boolean.
        - ``print_each`` is an integer ≥ 1.

    Args:
        p: The ``ModelParams`` instance to validate.

    Raises:
        ValueError: With a bullet-point list of every failing check.
    """
    errors: list[str] = []

    def _err(msg: str) -> None:
        errors.append(msg)

    # Architecture / identity -------------------------------------------------
    if not isinstance(p.name, str) or not p.name.strip():
        _err("'name' must be a non-empty string.")

    if not len(p.layer_sizes) >= 2:
        _err("number of layers must be ≥ 2}.")

    if not isinstance(p.layer_sizes, list):
        _err("'layer_sizes' must be a list.")
    else:
        for i, sz in enumerate(p.layer_sizes):
            if not isinstance(sz, int) or isinstance(sz, bool) or sz < 1:
                _err(f"'layer_sizes[{i}]' must be a positive integer, got {sz!r}.")

    # Training ----------------------------------------------------------------
    if not isinstance(p.learning_rate, (int, float)) or isinstance(p.learning_rate, bool) \
            or p.learning_rate <= 0:
        _err(f"'learning_rate' must be a positive number, got {p.learning_rate!r}.")

    if p.gradient_clip is not None:
        if not isinstance(p.gradient_clip, (int, float)) or isinstance(p.gradient_clip, bool) \
                or p.gradient_clip <= 0:
            _err(f"'gradient_clip' must be None or a positive number, got {p.gradient_clip!r}.")

    if not isinstance(p.loss_function, nn.Module):
        _err(
            f"'loss_function' must be an nn.Module instance (e.g. nn.MSELoss()), got {type(p.loss_function).__name__!r}.")

    if not isinstance(p.loss_function2, nn.Module):
        _err(f"'loss_function2' must be an nn.Module instance (e.g. nn.BCEWithLogitsLoss()), got {type(p.loss_function2).__name__!r}.")

    if p.tol is not None:
        if not isinstance(p.tol, (int, float)) or isinstance(p.tol, bool) or p.tol <= 0:
            _err(f"'tol' must be None or a positive number, got {p.tol!r}.")

    if not isinstance(p.max_epoch, int) or isinstance(p.max_epoch, bool) or p.max_epoch < 1:
        _err(f"'max_epoch' must be an integer ≥ 1, got {p.max_epoch!r}.")

    if not isinstance(p.activation_functions, list):
        _err("'activation_functions' must be a list.")
    else:
        expected = len(p.layer_sizes) - 2
        if len(p.activation_functions) != expected:
            _err(f"'activation_functions' must have exactly {expected} entries, got {len(p.activation_functions)}.")
        for i, fn in enumerate(p.activation_functions):
            if not isinstance(fn, nn.Module):
                _err(f"'activation_functions[{i}]' must be an nn.Module instance, got {type(fn).__name__!r}.")

    if p.seed is not None:
        if not isinstance(p.seed, int) or isinstance(p.seed, bool) or p.seed < 0:
            _err(f"'seed' must be None or a non-negative integer, got {p.seed!r}.")

    if not isinstance(p.shuffle, bool):
        _err(f"'shuffle' must be a boolean, got {type(p.shuffle).__name__!r}.")

    # Runtime / logging -------------------------------------------------------
    if not isinstance(p.device, str) or not p.device.strip():
        _err(f"'device' must be a non-empty string, got {p.device!r}.")

    if not isinstance(p.verbose, bool):
        _err(f"'verbose' must be a boolean, got {type(p.verbose).__name__!r}.")

    if not isinstance(p.print_each, int) or isinstance(p.print_each, bool) or p.print_each < 1:
        _err(f"'print_each' must be an integer ≥ 1, got {p.print_each!r}.")

    # Report all errors at once -----------------------------------------------
    if errors:
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"ModelParams validation failed with {len(errors)} error(s):\n{bullet_list}")