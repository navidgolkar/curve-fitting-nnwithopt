from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Seeding helper  (same interface as utils/parameters.py::apply_seed)
# ---------------------------------------------------------------------------

def apply_seed(seed: int | None) -> None:
    """
    Seed every source of randomness used by the Optimizers.

    Covers ``random`` and ``numpy``.  Does nothing when *seed* is ``None``.

    Args:
        seed: Non-negative integer seed, or ``None`` to skip seeding.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)

def _tanh(x: np.ndarray, a: float, lb: float, ub: float) -> np.ndarray:
    """
        hyperbolic tangent: (tanh(ax) + 1) / 2.

        Mathematically identical to the logistic sigmoid 1/(1+e^{-ax}) when a=0.5.
        Steeper for a > 1 (faster saturation); shallower for 0 < a < 1
        (more exploration-friendly).
        Numerically stable — no raw exp; uses tanh with input clipping.
        """
    
    return (np.tanh(np.clip(a * x, lb, ub)) + 1.0) / 2.0


def _softsign(x: np.ndarray, a: float) -> np.ndarray:
    """
    Softsign normalised to [0, 1]: (ax / (1 + a|x|) + 1) / 2.
    """
    return (a * x / (1.0 + a * np.abs(x)) + 1.0) / 2.0

def _hard_clip(x: np.ndarray, a: float) -> np.ndarray:
    """
    Piecewise-linear:

    Very cheap to evaluate; creates flat 0 and 1 plateaus that enforce strict
    keep/prune decisions far from the origin — good for aggressive pruning.
    Transition width scales with 1/a (larger a → sharper).
    """
    return np.clip(a*x + 0.5, 0.0, 1.0)


def _arctan(x: np.ndarray, a) -> np.ndarray:
    """
    Arctan normalised to [0, 1]:
    """
    return np.arctan(a * x) / np.pi + 0.5


def _gudermannian(x: np.ndarray, a: float, lb: float, ub: float) -> np.ndarray:
    """
    Gudermannian: (2/π) · arctan(tanh(ax/2)).

    A smooth, bounded S-curve with a steep central slope.  Useful for a
    moderate exploration–exploitation balance.  Output range is (-1, 1);
    here shifted and scaled to (0, 1) via the arctan mapping.
    """
    raw = (2.0 / np.pi) * np.arctan(np.tanh(np.clip(a * x / 2.0, lb, ub))) + 0.5
    return (raw + 1.0)/2.0


def _algebraic(x: np.ndarray, a: float) -> np.ndarray:
    """
    Algebraic normalized to [0, 1]: (ax / sqrt(1 + (ax)²) + 1) / 2.

    Intermediate saturation speed compared to hard_clip/softsign;
    fast and numerically robust on large arrays.
    """
    ax = a * x
    return (1.0 + ax / np.sqrt(1.0 + ax ** 2)) / 2.0

# Public dictionary -----------------------------------------------------------
NORMALIZE_DICT = {
    1: lambda x, a, lb=-30.0, ub=30.0: _tanh(x, a, lb, ub),
    2: lambda x, a: _softsign(x, a),
    3: lambda x, a: _hard_clip(x, a),
    4: lambda x, a: _arctan(x, a),
    5: lambda x, a, lb=-30.0, ub=30.0: _gudermannian(x, a, lb, ub),
    6: lambda x, a: _algebraic(x, a),
}
NORMALIZE_NAMES = {
    1: "tanh",
    2: "softsign",
    3: "hard_clip",
    4: "arctan",
    5: "gudermannian",
    6: "algebraic",
}

@dataclass
class GWOParams:
    """
    Configuration for the Binary Grey Wolf Optimizer.

    Attributes
    ----------
    Population
    ~~~~~~~~~~
    n_wolves : int
        Population size (number of search agents).  Must be ≥ 3 (need at
        least alpha, beta, delta leaders).
    max_iter : int
        Number of GWO iterations.  Must be ≥ 1.

    Search-space bias
    ~~~~~~~~~~~~~~~~~
    prune_ratio : float
        Target fraction of *inner* edges to prune.  Used only to bias the
        initial continuous positions toward pruning low-importance edges.
        Must be in (0, 1).

    Transfer function
    ~~~~~~~~~~~~~~~~~
    normalize_func : int
        Key into ``NORMALIZE_DICT`` selecting the function that maps
        continuous positions → keep-probabilities in [0, 1].
        Default 1 (tanh).
    normalize_a : float
        Steepness parameter *a* passed to the chosen transfer function.
        Must be > 0.  Default 1.0.

    Fitness shaping
    ~~~~~~~~~~~~~~~
    sparsity_bonus : float
        Small coefficient added to fitness per surviving inner edge, so that
        among topologies with equal convergence speed, sparser ones win.
        Must be ≥ 0.  Default 0.01.

    Misc
    ~~~~
    seed : int | None
        RNG seed applied to ``random`` and ``numpy`` before the GWO starts.
        ``None`` disables seeding.  When set, must be a non-negative integer.
    verbose : bool
        Print iteration summaries.  Default ``True``.
    print_each : int
        Print/log progress every *print_each* iterations.  Must be ≥ 1.

    Computed
    --------
    label : str  *(read-only property)*
        Short identifier: ``"GWO_w{n_wolves}_i{max_iter}_p{prune_ratio:.2f}"``.
    normalize_name : str  *(read-only property)*
        name of the normalization function.
    """
    
    # Population --------------------------------------------------------------
    n_wolves:   int   = 10
    max_iter:   int   = 20
    prune_ratio: float = 0.4
    
    # Transfer function -------------------------------------------------------
    normalize_func: int   = 1      # key into NORMALIZE_DICT
    normalize_a:    float = 1.0    # steepness parameter *a*

    # Misc --------------------------------------------------------------------
    seed:       int | None = None
    verbose:    bool       = False
    
    # Computed properties -----------------------------------------------------
    @property
    def label(self) -> str:
        return f"GWO_w{self.n_wolves}_i{self.max_iter}_p{self.prune_ratio:.2f}"
    
    @property
    def normalize_name(self) -> str:
        return NORMALIZE_NAMES.get(self.normalize_func, f"unknown_key")


def check_gwo_parameters(p: GWOParams) -> None:
    """
    Validate a ``GWOParams`` instance and raise ``ValueError`` listing *all*
    problems found (not just the first).

    Checks
    ------
    * ``n_wolves``       — integer ≥ 3.
    * ``max_iter``       — integer ≥ 1.
    * ``prune_ratio``    — float in (0, 1).
    * ``normalize_func`` — key present in ``NORMALIZE_DICT``.
    * ``normalize_a``    — positive number.
    * ``sparsity_bonus`` — non-negative number.
    * ``seed``           — ``None`` or non-negative integer.
    * ``verbose``        — boolean.
    * ``print_each``     — integer ≥ 1.

    Args:
        p: The ``GWOParams`` instance to validate.

    Raises:
        ValueError: With a bullet-point list of every failing check.
    """
    errors: list[str] = []
    
    def _err(msg: str) -> None:
        errors.append(msg)
    
    # n_wolves ----------------------------------------------------------------
    if not isinstance(p.n_wolves, int) or isinstance(p.n_wolves, bool) or p.n_wolves < 3:
        _err(f"'n_wolves' must be an integer ≥ 3 (alpha+beta+delta minimum), "
             f"got {p.n_wolves!r}.")
    
    # max_iter ----------------------------------------------------------------
    if not isinstance(p.max_iter, int) or isinstance(p.max_iter, bool) or p.max_iter < 1:
        _err(f"'max_iter' must be an integer ≥ 1, got {p.max_iter!r}.")
    
    # prune_ratio -------------------------------------------------------------
    if (not isinstance(p.prune_ratio, (int, float)) or isinstance(p.prune_ratio, bool) or not (
            0.0 < p.prune_ratio < 1.0)):
        _err(f"'prune_ratio' must be a float in (0, 1), got {p.prune_ratio!r}.")
    
    # normalize_func ----------------------------------------------------------
    if p.normalize_func not in NORMALIZE_DICT:
        _err(f"'normalize_func' must be one of {sorted(NORMALIZE_DICT.keys())}, "
             f"got {p.normalize_func!r}.")
    
    # normalize_a -------------------------------------------------------------
    if (not isinstance(p.normalize_a, (int, float)) or isinstance(p.normalize_a, bool) or p.normalize_a <= 0):
        _err(f"'normalize_a' must be a positive number, got {p.normalize_a!r}.")
    
    # seed --------------------------------------------------------------------
    if p.seed is not None:
        if not isinstance(p.seed, int) or isinstance(p.seed, bool) or p.seed < 0:
            _err(f"'seed' must be None or a non-negative integer, got {p.seed!r}.")
    
    # verbose -----------------------------------------------------------------
    if not isinstance(p.verbose, bool):
        _err(f"'verbose' must be a boolean, got {type(p.verbose).__name__!r}.")
        
    # Report all errors at once -----------------------------------------------
    if errors:
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        raise ValueError(f"GWOParams validation failed with {len(errors)} error(s):\n{bullet_list}")
