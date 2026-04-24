import numpy as np

class BinaryGWOptimizer: # Grey Wolves Optimizer
    """
    Binary Grey Wolves optimizer
    The GWO position update is done in continuous space; a sigmoid function
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
    S. Mirjalili, S.M. Mirjalili, A. Lewis (2014). Grey wolf optimizer.
    https://doi.org/10.1016/j.advengsoft.2013.12.007
    """
    # ------------------------------------------------------------------
    # Each Wolf
    # ------------------------------------------------------------------
    class Wolf:
        pos: np.ndarray | None
        score: float
        def __init__(self, pos: np.ndarray | None = None, score: float = np.inf):
            self.pos = pos
            self.score = score
    
    def __init__(self,
            n_wolves: int,
            n_edges: int,
            importance: np.ndarray,
            max_iter: int,
            prune_ratio: float = 0.4,
            seed: int | None = None):
        
        self.n_wolves = n_wolves
        self.n_edges = n_edges
        self.importance = importance    # shape = (n_edges,)
        self.prune_ratio = prune_ratio
        self.max_iter = max_iter
        
        self.rng = np.random.default_rng(seed if seed is not None else 42)
        
        # Continuous positions — initialized with importance bias
        self.positions = self._init_positions()
        
        # alpha, beta, and gamma wolves
        self.alpha = self.Wolf()
        self.beta = self.Wolf()
        self.delta = self.Wolf()
        
        self.score_history: list[float] = []
    
    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_positions(self) -> np.ndarray:
        """
        Continuous positions in (-3, 3)
        
        High-importance edges get a positive bias (push toward keep);
        low-importance edges get a negative bias (push toward prune).
        """
        base = (self.importance - 0.5) * 4.0  # scale to roughly (-2, 2)
        noise = self.rng.standard_normal((self.n_wolves, self.n_edges)) * 0.5
        return base[None, :] + noise  # broadcast: (n_wolves, n_edges) + noise
    
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))
    
    def to_binary(self, x: np.ndarray) -> np.ndarray:
        """Convert continuous positions to a binary keep (1) / prune (0) mask."""
        prob = self._sigmoid(x)
        return (0.5 < prob).astype(np.int8)
    
    def update_leaders(self, scores: np.ndarray) -> None:
        order = np.argsort(scores)
        if scores[order[0]] < self.alpha.score:
            self.alpha.score = scores[order[0]]
            self.alpha.pos = self.positions[order[0]].copy()
        if len(order) > 1 and scores[order[1]] < self.beta.score:
            self.beta.score = scores[order[1]]
            self.beta.pos = self.positions[order[1]].copy()
        if len(order) > 2 and scores[order[2]] < self.delta.score:
            self.delta.score = scores[order[2]]
            self.delta.pos = self.positions[order[2]].copy()
    
    # ------------------------------------------------------------------
    # Position update for one GWO iteration
    # ------------------------------------------------------------------
    def step(self, iteration: int) -> np.ndarray:
        """Update all wolf positions for one iteration; return binary masks."""
        a = 2.0 - 2.0*iteration/self.max_iter # linearly decreases from 2 to 0
        if self.alpha.pos is None:
            return np.array([self.to_binary(p) for p in self.positions])
        
        def _hunt(leader_pos: np.ndarray) -> np.ndarray:
            r1 = self.rng.random((self.n_wolves, self.n_edges))
            r2 = self.rng.random((self.n_wolves, self.n_edges))
            A = 2 * a * r1 - a
            C = 2 * r2
            D = np.abs(C * leader_pos - self.positions)
            return leader_pos - A * D
        
        X1 = _hunt(self.alpha.pos)
        X2 = _hunt(self.beta.pos)
        X3 = _hunt(self.delta.pos)
        
        self.positions = (X1 + X2 + X3) / 3.0
        
        return np.array([self.to_binary(p) for p in self.positions])
    
    