import torch
import torch.nn as nn
import numpy as np
import copy

from utils.parameters import ModelParams, apply_seed, check_parameters

# Connectivity helpers --------------------------------------------------------
def _full_connections(n_out: int, n_in: int) -> np.ndarray:
    """All-True (n_out × n_in) bool matrix: every output receives every input."""
    return np.ones((n_out, n_in), dtype=bool)
 
 
def _conv_connections(n_out: int, n_in: int, kernel_size: int, padding: int, stride: int) -> np.ndarray:
    """
    Sparse (n_out × n_in) connectivity matrix for Conv1d(n_in, n_out, ...).
    Entry [out_filter, in_filter] = True iff out_filter's receptive field
    overlaps with in_filter.
    """
    conn = np.zeros((n_out, n_in), dtype=bool)
    for j in range(n_out):
        i_min = j * stride - padding
        i_max = i_min + kernel_size - 1
        for i in range(n_in):
            if i_min <= i <= i_max:
                conn[j, i] = True
    return conn

# Fully Connected Neural Network ----------------------------------------------
class FCNN(nn.Module):
    """
    Fully-connected neural network built from ``params.layer_sizes``.
    
    Each hidden transition ``layer_sizes[i] → layer_sizes[i+1]`` is an
    ``nn.Linear`` followed by ``activation_functions[i]``.  The final
    hidden→output transition is a linear ``nn.Linear`` with no activation.
    
    Args:
        params : ModelParams
            * ``layer_sizes``         — width of every layer; can vary freely.
            * ``activation_functions``— ``len(layer_sizes) - 2`` activations,
              one per hidden transition.
            * ``seed``, ``device``, and all training fields.
    """
    def __init__(self, params: ModelParams):
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        sizes = params.layer_sizes
        funcs = params.activation_functions
        n_layers = len(sizes)
        
        # _connections[i]: (sizes[i+1] × sizes[i]) bool, one per transition
        params._connections = [_full_connections(sizes[i + 1], sizes[i]) for i in range(n_layers - 1)]
        
        layers: list[nn.Module] = []
        for i in range(n_layers - 2):
            layers.extend([nn.Linear(sizes[i], sizes[i + 1]), copy.deepcopy(funcs[i])])
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(sizes[-2], sizes[-1])
        
        params._is_resnet = False
        params._skip_connections = [[] for _ in range(n_layers)]
        
        self.to(params.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


# Convolutional Neural Network ------------------------------------------------
class CNN(nn.Module):
    """
    1-D convolutional neural network built from ``params.layer_sizes``.
 
    The first Conv1d uses ``kernel_size=1`` to fan out from ``layer_sizes[0]``
    input channels to ``layer_sizes[1]`` filters.  Subsequent hidden→hidden
    transitions use the supplied *kernel_size* / *padding* / *stride*.  The
    final Conv1d head collapses to ``layer_sizes[-1]`` output channels with
    ``kernel_size=1`` and no activation.
 
    Args:
        params      : ModelParams
        kernel_size : Conv1d kernel size for hidden→hidden transitions.
        padding     : Conv1d padding.
        stride      : Conv1d stride.
    """
    def __init__(self, params: ModelParams, kernel_size: int, padding: int, stride: int):
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        sizes = params.layer_sizes
        funcs = params.activation_functions
        n_layers = len(sizes)
        
        # _connections[i]: correct (out × in) shape for each transition.
        params._connections = [_full_connections(sizes[1], sizes[0])]
        for i in range(1, n_layers - 2):
            params._connections.append(_conv_connections(sizes[i + 1], sizes[i], kernel_size, padding, stride))
        params._connections.append(_full_connections(sizes[-1], sizes[-2]))
        
        # Hidden body
        conv_layers: list[nn.Module] = [nn.Conv1d(sizes[0], sizes[1], kernel_size=1), copy.deepcopy(funcs[0])]
        for i in range(1, n_layers-2):
            conv_layers.extend([nn.Conv1d(sizes[i], sizes[i + 1], kernel_size=kernel_size, padding=padding, stride=stride), copy.deepcopy(funcs[i])])
        self.conv = nn.Sequential(*conv_layers)
        self.head = nn.Conv1d(sizes[-2], sizes[-1], kernel_size=1)
        
        params._is_resnet = False
        params._skip_connections = [[] for _ in range(n_layers)]
        
        self.to(params.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Instead of permute you can Transpose (x.T) but to be consistent with ConvResNet which needs permute I used permute here as well
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x   = x.permute(0, 2, 1)       # (B, sizes[0], N)
        out = self.conv(x)
        out = self.head(out)
        out = out.permute(0, 2, 1)      # (B, N, sizes[-1])
        return out.squeeze(0)

# Dense Residual Network ------------------------------------------------------
class DenseResNet(nn.Module):
    """
    Dense Residual Network built from ``params.layer_sizes``.
 
    Every hidden layer sends a parameter-free residual to every later hidden
    layer (full layer index i → j for all j > i + 1, i ≥ 1).
    All *hidden* layers must share the same width.
 
    Args:
        params : ModelParams
    """
    def __init__(self, params: ModelParams):
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        sizes = params.layer_sizes
        funcs = params.activation_functions
        self.n_layers = len(sizes)
        
        if len(set(sizes[1:-1])) > 1:
            raise ValueError(f"DenseResNet requires uniform hidden widths, got {sizes[1:-1]}.")
        
        # _connections: full transitions including head
        params._connections = [_full_connections(sizes[i + 1], sizes[i]) for i in range(self.n_layers - 1)]
        
        # Skip-connection maps — full layer indices, length = n_layers
        self._skip_sources: list[list[int]] = [[] for _ in range(self.n_layers)]
        for j in range(1, self.n_layers - 1):       # j: hidden layers in full-layer index
            for i in range(1, j - 1):           # i: all earlier hidden layers (skip gap > 1)
                self._skip_sources[j].append(i)
        
        # Hidden body
        layer_list: list[nn.Module] = [nn.Sequential(nn.Linear(sizes[0], sizes[1]), copy.deepcopy(funcs[0]))]
        
        for i in range(1, self.n_layers - 2):
            layer_list.append(nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), copy.deepcopy(funcs[i])))
        self.layers = nn.ModuleList(layer_list)
        self.head   = nn.Linear(sizes[-2], sizes[-1])
        
        params._is_resnet = True
        params._skip_connections = self._skip_sources
        
        self.to(params.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        layer_input = x
        for idx in range(self.n_layers - 2):
            outputs.append(self.layers[idx](layer_input))
            layer_input = outputs[-1]
            for src_idx in self._skip_sources[idx+1]:
                layer_input = layer_input + outputs[src_idx-1]
        return self.head(layer_input)

# Convolutional Residual Network ----------------------------------------------
class ConvResNet(nn.Module):
    """
    Convolutional Residual Network built from ``params.layer_sizes``.
 
    Skip-connection rule (full layer indices, hidden layers only):
        Hidden layer i sends a skip to j where i+1 < j ≤ i+connect+1,
        only when i + connect + 2 ≤ n_layers - 2.
    All *hidden* layers must share the same filter count.
 
    Args:
        params      : ModelParams
        kernel_size : Conv1d kernel size for hidden→hidden transitions.
        padding     : Conv1d padding.
        stride      : Conv1d stride.
        connect     : skip window size.  ``connect=0`` → no skips.
    """
    def __init__(self, params: ModelParams, kernel_size: int, padding: int, stride: int, connect: int = 1):
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.connect = connect
        
        sizes = params.layer_sizes
        funcs = params.activation_functions
        self.n_layers = len(sizes)
        
        if len(set(sizes[1:-1])) > 1:
            raise ValueError(f"ConvResNet requires uniform hidden filter counts, got {sizes[1:-1]}.")
        
        # _connections[i]: correct (out × in) for each transition.
        params._connections = [_full_connections(sizes[1], sizes[0])]
        for i in range(1, self.n_layers - 2):
            params._connections.append(_conv_connections(sizes[i + 1], sizes[i], kernel_size, padding, stride))
        params._connections.append(_full_connections(sizes[-1], sizes[-2]))
        
        # Skip-connection maps — full layer indices, length = n_layers
        h_n = self.n_layers - 2   # number of hidden layers
        self._skip_targets: list[list[int]] = [[] for _ in range(self.n_layers)]
        for hi in range(h_n):             # hi: hidden-local source
            i = hi + 1                    # i: full layer index
            if hi + connect + 2 <= h_n:
                for hj in range(h_n):
                    if hi + 1 < hj <= hi + connect + 1:
                        self._skip_targets[i].append(hj + 1)
            
        self._skip_sources: list[list[int]] = [[] for _ in range(self.n_layers)]
        for i, targets in enumerate(self._skip_targets):
            for j in targets:
                self._skip_sources[j].append(i)
                
        # Hidden Body
        layer_list: list[nn.Module] = [nn.Sequential(nn.Conv1d(sizes[0], sizes[1], kernel_size=1), copy.deepcopy(funcs[0]))]
        for i in range(1, self.n_layers - 2):
            layer_list.append(nn.Sequential(nn.Conv1d(sizes[i], sizes[i + 1], kernel_size=kernel_size, padding=padding, stride=stride), copy.deepcopy(funcs[i])))
        self.layers = nn.ModuleList(layer_list)
        self.head = nn.Conv1d(sizes[-2], sizes[-1], kernel_size=1)
        
        params._is_resnet = True
        params._skip_connections = self._skip_sources
        
        self.to(params.device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.permute(0, 2, 1)       # (B, sizes[0], N)
        outputs: list[torch.Tensor] = []
        layer_input = x
        for idx in range(self.n_layers - 2):
            outputs.append(self.layers[idx](layer_input))
            layer_input = outputs[-1]
            for src_idx in self._skip_sources[idx+1]:
                layer_input = layer_input + outputs[src_idx-1]
        out = self.head(layer_input)
        out = out.permute(0, 2, 1)      # (B, N, sizes[-1])
        return out.squeeze(0)
    
# Custom Graph Network --------------------------------------------------------
class CustomNet(nn.Module):
    """
    Arbitrary feedforward network defined by an explicit node-connection graph.
 
    **Adjacent edges** (``tgt_layer == src_layer + 1``) pass through a masked
    ``nn.Linear``.  Inactive connections are zeroed and stay zeroed.
    **Skip edges** (``tgt_layer > src_layer + 1``) are parameter-free additions.
 
    Args:
        params : ModelParams
        nodes  : flat list of ``(src_layer, src_node, tgt_layer, tgt_node)``
            tuples defining every active edge.  The list need not be sorted.
    """
    def __init__(self, params: ModelParams, nodes: list[tuple[int, int, int, int]]) -> None:
        super().__init__()
        check_parameters(params)
        apply_seed(params.seed)
        
        self.params = params
        self.layer_sizes = list(params.layer_sizes)
        self.n_layers = len(self.layer_sizes)
        funcs = params.activation_functions
        
        # Parse nodes ---------------------------------------------------------
        adjacent_active: dict[int, set[tuple[int, int]]] = {}
        skip_pairs: set[tuple[int, int]] = set()
        
        for src_layer, src_node, tgt_layer, tgt_node in nodes:
            self.check_connection(src_layer, src_node, tgt_layer, tgt_node)
            if tgt_layer == src_layer + 1:
                adjacent_active.setdefault(src_layer, set()).add((tgt_node, src_node))
            else:
                skip_pairs.add((src_layer, tgt_layer))
        
        # The full dense graph is created first; edges absent from nodes are
        # then pruned so their weights are zeroed and kept at zero by
        # _apply_masks on every forward pass.
        self._linears = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1], bias=True) for i in range(self.n_layers - 1)])
        
        # _connections: start all-True (full dense), then mark inactive False
        params._connections = [_full_connections(self.layer_sizes[i + 1], self.layer_sizes[i]) for i in range(self.n_layers - 1)]
        for i in range(self.n_layers - 1):
            active_set = adjacent_active.get(i, set())
            for tgt_n in range(self.layer_sizes[i + 1]):
                for src_n in range(self.layer_sizes[i]):
                    if (tgt_n, src_n) not in active_set:
                        params._connections[i][tgt_n, src_n] = False
        
        # Active masks derived from _connections (registered as buffers)
        self._active_mask: list[torch.Tensor] = []
        for i in range(self.n_layers - 1):
            mask = torch.from_numpy(params._connections[i].copy())
            self.register_buffer(f"_mask_{i}", mask)
            self._active_mask.append(getattr(self, f"_mask_{i}"))
        
        # Skip maps -----------------------------------------------------------
        self._skip_targets: list[list[int]] = [[] for _ in range(self.n_layers)]
        self._skip_sources: list[list[int]] = [[] for _ in range(self.n_layers)]
        for (src_l, tgt_l) in skip_pairs:
            self._skip_targets[src_l].append(tgt_l)
            self._skip_sources[tgt_l].append(src_l)
        for lst in self._skip_targets:
            lst.sort()
        for lst in self._skip_sources:
            lst.sort()
        
        # Activations: funcs covers hidden transitions; identity for output
        self._funcs = nn.ModuleList([copy.deepcopy(f) for f in funcs] + [nn.Identity()])
        
        # Pruning -------------------------------------------------------------
        # Zero all inactive adjacent edges immediately
        self._pruned: set[tuple[int, int, int, int]] = set()
        self._apply_masks()
        # Apply any edges pre-declared pruned in params
        for edge in params._pruned:
            self.add_pruned(*edge)
        
        # Write back to params ------------------------------------------------
        params._is_resnet = bool(skip_pairs)
        params._skip_connections = self._skip_sources
        
        self.to(params.device)
    
    def _apply_masks(self) -> None:
        with torch.no_grad():
            for lin, mask in zip(self._linears, self._active_mask):
                lin.weight.data[~mask] = 0.0
                dead = ~mask.any(dim=1)
                if dead.any():
                    lin.bias.data[dead] = 0.0
    
    def add_pruned(self, src_layer: int, src_node: int, tgt_layer: int, tgt_node: int) -> None:
        """Mark an edge as pruned, updating weights, masks, and _connections."""
        self.check_connection(src_layer, src_node, tgt_layer, tgt_node)
        edge = (src_layer, src_node, tgt_layer, tgt_node)
        self._pruned.add(edge)
        self.params._pruned.add(edge)
        
        if tgt_layer == src_layer + 1:
            # Zero weight and update mask
            mask = self._active_mask[src_layer]
            mask[tgt_node, src_node] = False
            with torch.no_grad():
                lin = self._linears[src_layer]
                lin.weight.data[tgt_node, src_node] = 0.0
                if not mask[tgt_node].any():
                    lin.bias.data[tgt_node] = 0.0
            # Mirror into params._connections
            self.params._connections[src_layer][tgt_node, src_node] = False
        else:
            # Remove from skip routing maps
            if tgt_layer in self._skip_targets[src_layer]:
                self._skip_targets[src_layer].remove(tgt_layer)
            if src_layer in self._skip_sources[tgt_layer]:
                self._skip_sources[tgt_layer].remove(src_layer)
            self.params._skip_connections = self._skip_sources

    def check_connection(self, src_layer: int, src_node: int, tgt_layer: int, tgt_node: int) -> None:
        """Validate a connection tuple (src_layer, src_node, tgt_layer, tgt_node)."""
        # Layer checks -----------------------------------------------------------------
        if src_layer >= self.n_layers-1 or src_layer < 0:
            raise ValueError(f"src_layer {src_layer} out of range [0, {self.n_layers-1}].")
        if tgt_layer >= self.n_layers or tgt_layer < 0:
            raise ValueError(f"tgt_layer {tgt_layer} out of range [0, {self.n_layers-1}].")
        if tgt_layer <= src_layer:
            raise ValueError(f"Invalid connection ({src_layer}->{tgt_layer}): must satisfy tgt_layer > src_layer")
        # Node checks ------------------------------------------------------------------
        if src_node < 0 or src_node >= self.layer_sizes[src_layer]:
            raise ValueError(f"src_node {src_node} out of range for layer {src_layer} (size {self.layer_sizes[src_layer]})")
        if tgt_node < 0 or tgt_node >= self.layer_sizes[tgt_layer]:
            raise ValueError(f"tgt_node {tgt_node} out of range for layer {tgt_layer} (size {self.layer_sizes[tgt_layer]})")
        # Skip connection check --------------------------------------------------------
        if tgt_layer > src_layer + 1 and self.layer_sizes[src_layer] != self.layer_sizes[tgt_layer]:
            raise ValueError(f"Skip edge ({src_layer})→({tgt_layer}) requires equal layer sizes ({self.layer_sizes[src_layer]} != {self.layer_sizes[tgt_layer]})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._apply_masks()
        outputs: list[torch.Tensor] = []
        layer_input = x
        for L in range(self.n_layers-1):
            outputs.append(self._funcs[L](self._linears[L](layer_input)))
            # calculating input of next layer which is the output of current(L) layer
            layer_input = outputs[-1]
            for src_l in self._skip_sources[L+1]:
                layer_input = layer_input + outputs[src_l-1]
        return outputs[-1]