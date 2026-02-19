import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, TransformerConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops


class LinkPredictor(nn.Module):
    def __init__(self, hidden_dim, mlp_layers=2, mlp_hidden=512, dropout=0.5):
        super().__init__()
        layers = []
        inp = hidden_dim
        for i in range(mlp_layers):
            layers.append(nn.Linear(inp, mlp_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(mlp_hidden))
            layers.append(nn.Dropout(dropout))
            inp = mlp_hidden
        layers.append(nn.Linear(inp, 1))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight'): nn.init.ones_(m.weight)
                if hasattr(m, 'bias'): nn.init.zeros_(m.bias)

    def forward(self, z, edge_pairs):
        if isinstance(edge_pairs, tuple):
            u_idx, v_idx = edge_pairs[0], edge_pairs[1]
        elif edge_pairs.dim() == 2 and edge_pairs.size(0) == 2:
            u_idx, v_idx = edge_pairs[0], edge_pairs[1]
        else:
            u_idx, v_idx = edge_pairs[:, 0], edge_pairs[:, 1]

        h_u = z[u_idx]
        h_v = z[v_idx]
        x = h_u * h_v
        logits = self.mlp(x).squeeze(-1)
        return logits


# === MPNN Architecture ===
class SimpleMPNNLayer(MessagePassing):
    def __init__(self, hidden_dim, msg_hidden=None):
        super().__init__(aggr='mean')
        msg_hidden = msg_hidden or hidden_dim
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, msg_hidden), nn.ReLU(),
            nn.Linear(msg_hidden, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def reset_parameters(self):
        for m in list(self.msg_mlp) + list(self.update_mlp):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, edge_index):
        if not isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        edge_index = edge_index.to(x.device)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        m = self.propagate(edge_index, x=x)
        out = self.update_mlp(x + m)
        return out

    def message(self, x_j):
        return self.msg_mlp(x_j)


# === HLMPNN ===
class HLMPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=10, msg_hidden=128, use_activation=False, beta_mode='learnable'):
        super().__init__()
        self.num_layers = num_layers
        self.input_lin = nn.Linear(in_channels, hidden_channels)
        self.mpns = nn.ModuleList([SimpleMPNNLayer(hidden_channels, msg_hidden) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        # --- BETA ABLATION LOGIC ---
        self.beta_mode = beta_mode
        if beta_mode == 'learnable':
            self.beta = nn.Parameter(torch.randn(num_layers + 1))
        elif beta_mode == 'uniform':
            self.register_buffer('beta', torch.ones(num_layers + 1))
        elif beta_mode == 'exponential':
            # Decay: 1, 0.5, 0.25, 0.125...
            weights = torch.tensor([0.5 ** i for i in range(num_layers + 1)])
            self.register_buffer('beta', weights)
        else:
            raise ValueError(f"Unknown beta_mode: {beta_mode}")

        self.use_activation = use_activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_lin.weight)
        if self.input_lin.bias is not None: nn.init.zeros_(self.input_lin.bias)
        for mp in self.mpns: mp.reset_parameters()
        for norm in self.norms: norm.reset_parameters()

        # Only reset if it is a learnable parameter
        if self.beta_mode == 'learnable':
            nn.init.normal_(self.beta, mean=0.0, std=0.01)

    def forward(self, x, edge_index, pe=None):
        if self.training:
            edge_index, _ = gcn_norm(edge_index, num_nodes=x.size(0), add_self_loops=True)

        z_prev = self.input_lin(x)
        if self.use_activation: z_prev = F.relu(z_prev)
        zs = [z_prev]

        for l in range(self.num_layers):
            z_l = self.mpns[l](z_prev, edge_index)
            z_l = self.norms[l](z_l)
            if self.use_activation: z_l = F.relu(z_l)
            zs.append(z_l)
            z_prev = z_l

        # Handle Normalization based on mode
        if self.beta_mode == 'learnable':
            betas = F.softmax(self.beta, dim=0)
        else:
            betas = self.beta / self.beta.sum()

        return sum(b * z for b, z in zip(betas, zs))


# === HLAttention ===
class HLAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=10, heads=2, att_dropout=0.3, layer_dropout=0.1,
                 use_activation=True, beta_mode='learnable'):
        super().__init__()
        assert hidden_channels % heads == 0, "Hidden channels must be divisible by heads"
        self.num_layers = num_layers
        self.layer_dropout = layer_dropout
        self.use_activation = use_activation

        self.input_lin = nn.Linear(in_channels, hidden_channels)
        out_per_head = hidden_channels // heads

        self.gats = nn.ModuleList([
            GATConv(hidden_channels, out_per_head, heads=heads, concat=True, dropout=att_dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])

        # --- SAFE BETA INITIALIZATION ---
        self.beta_mode = beta_mode

        if beta_mode == 'learnable':
            self.beta = nn.Parameter(torch.randn(num_layers + 1))
        elif beta_mode == 'uniform':
            self.register_buffer('beta', torch.ones(num_layers + 1))
        elif beta_mode == 'exponential':
            weights = torch.tensor([0.5 ** i for i in range(num_layers + 1)])
            self.register_buffer('beta', weights)
        else:
            # This forces a crash early if the mode is wrong
            raise ValueError(f"Invalid beta_mode: '{beta_mode}'. Must be 'learnable', 'uniform', or 'exponential'.")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_lin.weight)
        if self.input_lin.bias is not None: nn.init.zeros_(self.input_lin.bias)
        for gat, norm in zip(self.gats, self.norms):
            gat.reset_parameters()
            norm.reset_parameters()

        # Only init if parameter exists
        if self.beta_mode == 'learnable':
            nn.init.normal_(self.beta, mean=0.0, std=0.01)

    def forward(self, x, edge_index, pe=None):
        z_prev = self.input_lin(x)
        if self.use_activation: z_prev = F.relu(z_prev)
        zs = [z_prev]

        for l in range(self.num_layers):
            if self.training and torch.rand(1).item() < self.layer_dropout:
                zs.append(z_prev)
                continue

            z_l = self.gats[l](z_prev, edge_index)
            z_l = self.norms[l](z_l)

            if self.use_activation: z_l = F.relu(z_l)

            z_l = z_l + z_prev
            zs.append(z_l)
            z_prev = z_l

        # Handle Normalization
        if self.beta_mode == 'learnable':
            betas = F.softmax(self.beta, dim=0)
        else:
            # This line caused your error because self.beta didn't exist
            betas = self.beta / self.beta.sum()

        return sum(b * z for b, z in zip(betas, zs))

# === HLGT ===
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class HLGT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=6, heads=2, att_dropout=0.3, drop_path=0.1, pe_dim=0,
                 use_activation=True, beta_mode='learnable'):
        super().__init__()
        self.num_layers = num_layers
        self.use_activation = use_activation

        self.input_lin = nn.Linear(in_channels + pe_dim, hidden_channels)
        out_per_head = hidden_channels // heads

        self.layers = nn.ModuleList(
            [TransformerConv(hidden_channels, out_per_head, heads=heads, dropout=att_dropout) for _ in
             range(num_layers)])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2), nn.GELU(), nn.Dropout(att_dropout),
                nn.Linear(hidden_channels * 2, hidden_channels)
            )
            for _ in range(num_layers)
        ])

        self.norm1 = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(hidden_channels) for _ in range(num_layers)])
        self.drop_paths = nn.ModuleList(
            [DropPath(drop_path) if drop_path > 0 else nn.Identity() for _ in range(num_layers)])

        # --- BETA ABLATION LOGIC ---
        self.beta_mode = beta_mode
        if beta_mode == 'learnable':
            self.beta = nn.Parameter(torch.randn(num_layers + 1))
        elif beta_mode == 'uniform':
            self.register_buffer('beta', torch.ones(num_layers + 1))
        elif beta_mode == 'exponential':
            weights = torch.tensor([0.5 ** i for i in range(num_layers + 1)])
            self.register_buffer('beta', weights)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_lin.weight)
        if self.input_lin.bias is not None: nn.init.zeros_(self.input_lin.bias)
        for layer in self.layers: layer.reset_parameters()
        for ffn in self.ffns:
            for m in ffn:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        if self.beta_mode == 'learnable':
            nn.init.normal_(self.beta, mean=0.0, std=0.01)

    def forward(self, x, edge_index, pe=None):
        if pe is not None:
            x = torch.cat([x, pe], dim=-1)

        z_prev = F.relu(self.input_lin(x))
        zs = [z_prev]

        for l in range(self.num_layers):
            attn_out = self.layers[l](z_prev, edge_index)
            attn_out = self.norm1[l](z_prev + self.drop_paths[l](attn_out))

            ffn_out = self.ffns[l](attn_out)
            z_l = self.norm2[l](attn_out + self.drop_paths[l](ffn_out))

            if self.use_activation: z_l = F.relu(z_l)
            zs.append(z_l)
            z_prev = z_l

        # Handle Normalization
        if self.beta_mode == 'learnable':
            betas = F.softmax(self.beta, dim=0)
        else:
            betas = self.beta / self.beta.sum()

        return sum(b * z for b, z in zip(betas, zs))