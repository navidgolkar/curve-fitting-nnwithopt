from __future__ import annotations
import numpy as np
from numpy.ma.core import empty_like

from optimizer.opt_parameters import (
    GWOParams,
    NORMALIZE_DICT,
    apply_seed,
    check_gwo_parameters,
)


class BinaryGWOptimizer: # Gray Wolf Optimizer
    """
    Binary Grey Wolves optimizer
    The GWO position update is done in continuous space; a tanh function
    maps positions to probabilities, and a deterministic threshold (prob > 0.5)
    gives the binary keep/prune mask.
    A weight-importance bias is applied at initialization and also shifts the
    probability toward keeping high-importance edges.
    The fitness function minimizes the number of training epochs required to
    reach the tolerance threshold ``tol``.  Wolves that produce a graph where
    training never reaches ``tol`` within ``max_epoch`` receive a penalty equal
    to ``max_epoch``.  An additional sparsity bonus encourages sparser graphs
    among solutions that converge in the same number of epochs.
    Edges that connect to/from the input layer (layer 0) or the output layer
    (layer ``n_layers - 1``) are **never pruned** — the optimizer receives only
    the *inner* edges as its search space.
    
    References
    ----------
    S. Mirjalili, S.M. Mirjalili, A. Lewis (2014). Grey Wolf Optimizer.
    https://doi.org/10.1016/j.advengsoft.2013.12.007
    """
    # ------------------------------------------------------------------
    # Each Wolf
    # ------------------------------------------------------------------
    class Wolf:
        pos: np.ndarray | None
        score: float
        def __init__(self, pos: np.ndarray | None = None, score: float = float("inf")):
            self.pos = pos
            self.score = score
    
    def __init__(self, gwo_params: GWOParams, n_edges: int, importance: np.ndarray,):
        
        check_gwo_parameters(gwo_params)
        self.p = gwo_params
        self.n_edges = n_edges
        self.importance = importance    # shape = (n_edges,)
        
        apply_seed(gwo_params.seed)
        self.rng = np.random.default_rng(gwo_params.seed)
        
        # Continuous positions — initialized with importance bias
        self.positions = self._init_positions()
        
        # Pack leaders
        self.alpha = self.Wolf()
        self.beta = self.Wolf()
        self.delta = self.Wolf()
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_positions(self) -> np.ndarray:
        """
        Continuous positions in (-3, 3)
        
        importance ≈ 1  →  positive bias  →  normalize(pos) > 0.5  →  *keep*
        importance ≈ 0  →  negative bias  →  normalize(pos) < 0.5  →  *prune*
 
        The prune_ratio shifts the baseline: a ratio of 0.4 pushes 40% of
        edges toward the prune side regardless of importance.

        """
        # noise = self.rng.standard_normal((self.p.n_wolves, self.n_edges)) + (0.5 - self.p.prune_ratio) * 6.0
        # return noise
        def low_interp(x: float, c: float) -> float:
            return (x - c)*3.0/c
        def high_interp(x: float, c: float) -> float:
            return (x - c)*3.0/(1.0 - c)
        noise = self.rng.standard_normal((self.p.n_wolves, self.n_edges))
        pos = np.empty_like(noise)
        for i in range(self.p.n_wolves):
            blended = np.clip((self.importance + noise[i])/2, 0, 1)
            for j in range(self.n_edges):
                pos[i][j] = low_interp(blended[j], self.p.prune_ratio) if blended[j] < self.p.prune_ratio else high_interp(blended[j], self.p.prune_ratio)
        return pos
    
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def to_binary(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the chosen transfer function to get keep-probabilities.
        Then convert continuous positions to binary keep (1) / prune (0) mask.
        
        Deterministic threshold at 0.5 (equivalent to sign of position for
        symmetric transfer functions like tanh centred at 0).
        """
        fn = NORMALIZE_DICT[self.p.normalize_func]
        return (fn(x, self.p.normalize_a) >= 0.5).astype(np.int8)
    
    def update_leaders(self, scores: np.ndarray) -> None:
        """
        Refresh alpha / beta / delta from the current population.
        Cascading update:
            - new alpha  → old alpha demoted to beta  → old beta demoted to delta
            - otherwise check beta and delta independently.
        Parameters
        ----------
        scores : 1-D array of shape ``(n_wolves,)`` — fitness per wolf
                (lower is better — fewer epochs to converge).
        """
        for i in range(self.p.n_wolves):
            fitness = scores[i]
            pos = self.positions[i]
            
            if fitness < self.alpha.score:
                # Cascade: alpha → beta → delta
                self.delta.score = self.beta.score
                self.delta.pos = self.beta.pos.copy() if self.beta.pos is not None else None
                self.beta.score = self.alpha.score
                self.beta.pos = self.alpha.pos.copy() if self.alpha.pos is not None else None
                self.alpha.score = fitness
                self.alpha.pos = pos.copy()
                
            elif self.alpha.score < fitness < self.beta.score:
                self.delta.score = self.beta.score
                self.delta.pos   = self.beta.pos.copy() if self.beta.pos is not None else None
                self.beta.score  = fitness
                self.beta.pos    = pos.copy()
                
            elif self.alpha.score < fitness and self.beta.score < fitness < self.delta.score:
                self.delta.score = fitness
                self.delta.pos   = pos.copy()
    
    # ------------------------------------------------------------------
    # Position update for one GWO iteration
    # ------------------------------------------------------------------
    def step(self, iteration: int) -> np.ndarray:
        """
        Update all wolf positions for one iteration; return binary masks.
        per-wolf, per-dimension loop.  Before leaders are elected, the raw
        (biased) positions are returned as binary masks.
 
        Parameters
        ----------
        iteration : current 0-based iteration index.
 
        Returns
        -------
        masks : int8 array of shape ``(n_wolves, n_edges)``
                1 = keep edge, 0 = prune edge.

        """
        # 'a' linearly decreases from 2 to 0
        a = 2.0 - 2.0*iteration/self.p.max_iter
        
        if self.alpha.pos is None:
            # No leaders yet — just threshold current positions
            return np.array([self.to_binary(p) for p in self.positions])
        
        new_positions = np.empty_like(self.positions)
        
        def _hunt(leader_pos: np.ndarray, wolf_pos: np.ndarray) -> np.ndarray:
            r1 = self.rng.random()
            r2 = self.rng.random()
            a_hunt = 2.0 * a * r1 - a
            c_hunt = 2.0 * r2
            D = np.abs(c_hunt * leader_pos - wolf_pos)
            return leader_pos - a_hunt * D
        
        for i in range(self.p.n_wolves):
            X1 = _hunt(self.alpha.pos, self.positions[i])
            beta = self.beta.pos if self.beta.pos is not None else self.alpha.pos
            X2 = _hunt(beta, self.positions[i])
            delta = self.delta.pos if self.delta.pos is not None else self.alpha.pos
            X3 = _hunt(delta, self.positions[i])
            new_positions[i] = np.clip((X1 + X2 + X3) / 3.0, -6.0, 6.0)
        
        self.positions = new_positions
        return np.array([self.to_binary(p) for p in self.positions])