import e3nn.nn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.gate_points_2101 import Convolution
from torch_geometric.nn import GATv2Conv, GINEConv, PNAConv, TransformerConv, GPSConv
from torch_geometric.nn.models.basic_gnn import MLP
from torch_geometric.nn.norm import BatchNorm, GraphNorm, LayerNorm, InstanceNorm, GraphSizeNorm, PairNorm, MeanSubtractionNorm
from torch_geometric.nn import radius_graph
from torch_geometric.utils import degree


class SE3ScoreModel(nn.Module):
    def __init__(self, hidden_dim=56, t_embed_dim=32, num_neighbors=10, radius=1.0, num_basis=16, lmax=3):
        super().__init__()

        self.radius = radius
        self.num_basis = num_basis
        self.num_neighbors = num_neighbors
        self.lmax = lmax

        # Time embedding MLP
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim)
        )

        # Define irreps
        irreps_node_input = Irreps(f"{1}x0e")  # 1 for ones + hidden_dim from t_embed
        irreps_node_attr = Irreps(f"{t_embed_dim}x0e")  # Node attributes
        irreps_edge_attr = Irreps.spherical_harmonics(lmax=self.lmax)  # "1o"
        irreps_hidden = Irreps(f"{hidden_dim//2}x0e + {hidden_dim//8}x1e + {hidden_dim//2}x1o + {hidden_dim//8}x2o")
        irreps_output = Irreps("1o")  # Vectorial output (3-dimensional)

        self.conv1 = Convolution(
            irreps_in=irreps_node_input,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_out=irreps_hidden,
            number_of_basis=num_basis + t_embed_dim,
            radial_layers=1,
            radial_neurons=hidden_dim,
            num_neighbors=num_neighbors
        )

        self.conv2 = Convolution(
            irreps_in=irreps_hidden,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_out=irreps_hidden,
            number_of_basis=num_basis + t_embed_dim,
            radial_layers=1,
            radial_neurons=hidden_dim,
            num_neighbors=num_neighbors
        )

        self.conv3 = Convolution(
            irreps_in=irreps_hidden,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_out=irreps_hidden,
            number_of_basis=num_basis + t_embed_dim,
            radial_layers=1,
            radial_neurons=hidden_dim,
            num_neighbors=num_neighbors
        )

        self.conv4 = Convolution(
            irreps_in=irreps_hidden,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_out=irreps_output,
            number_of_basis=num_basis + t_embed_dim,
            radial_layers=1,
            radial_neurons=hidden_dim,
            num_neighbors=num_neighbors
        )

        self.mlp1 = nn.Linear(t_embed_dim, t_embed_dim)
        self.mlp2 = nn.Linear(t_embed_dim, t_embed_dim)
        self.mlp3 = nn.Linear(t_embed_dim, t_embed_dim)
        self.mlp4 = nn.Linear(t_embed_dim, t_embed_dim)

        self.t_act = nn.SiLU()
        self.norm_act = e3nn.nn.NormActivation(irreps_hidden, torch.nn.SiLU(), normalize=True)

    def forward(self, coords, batch, t):
        # Time embedding
        t_embed = self.t_act(self.t_embed(t.unsqueeze(-1)))

        # Node features
        node_input = torch.ones(coords.shape[0], 1, device=coords.device)

        # Edge attributes
        edge_src, edge_dst = radius_graph(
            coords, r=self.radius, batch=batch, max_num_neighbors=self.num_neighbors
        )
        edge_vec = coords[edge_dst] - coords[edge_src]

        edge_sh = spherical_harmonics(
            l=np.arange(0, self.lmax+1).tolist(), x=edge_vec, normalize=True, normalization='component'
        )  # [num_edges, sum l]

        edge_length = edge_vec.norm(dim=1)  # [num_edges]

        # Radial embedding
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.radius,
            number=self.num_basis,  # match convolution's number_of_basis parameter
            basis='gaussian',
            cutoff=True
        )  # [num_edges, number_of_basis]

        # Compute node attribute (same as t embedding) and adding t_feat[edge_batch] to edge length embedding
        t_feat = self.mlp1(t_embed)
        node_attr = t_feat[batch]  # t per node: [N, t_embed_dim]
        edge_time = t_feat[batch[edge_src]]  # t per edge: [num_edges, t_embed_dim]

        # Convolution layers
        node_hidden1 = self.conv1(
            node_input=node_input,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh, # angular info
            edge_length_embedded=torch.cat([edge_length_embedded, edge_time], dim=1) # dist info
        )
        node_hidden1 = self.norm_act(node_hidden1)

        t_feat = self.mlp2(t_embed)
        node_attr = t_feat[batch]  # t per node: [N, t_embed_dim]
        edge_time = t_feat[batch[edge_src]]  # t per edge: [num_edges, t_embed_dim]

        node_hidden2 = self.conv2(
            node_input=node_hidden1,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_length_embedded=torch.cat([edge_length_embedded, edge_time], dim=1)
        )
        node_hidden2 = self.norm_act(node_hidden2)
        node_hidden2 = node_hidden2 + node_hidden1  # residual structure

        t_feat = self.mlp3(t_embed)
        node_attr = t_feat[batch]  # t per node: [N, t_embed_dim]
        edge_time = t_feat[batch[edge_src]]  # t per edge: [num_edges, t_embed_dim]

        node_hidden3 = self.conv3(
            node_input=node_hidden2,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_length_embedded=torch.cat([edge_length_embedded, edge_time], dim=1)
        )
        node_hidden3 = self.norm_act(node_hidden3)
        node_hidden3 = node_hidden3 + node_hidden2 # residual structure

        t_feat = self.mlp4(t_embed)
        node_attr = t_feat[batch]  # t per node: [N, t_embed_dim]
        edge_time = t_feat[batch[edge_src]]  # t per edge: [num_edges, t_embed_dim]

        node_output = self.conv4(
            node_input=node_hidden3,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_length_embedded=torch.cat([edge_length_embedded, edge_time], dim=1)
        )

        return node_output


class GATv2ScoreModel(nn.Module):
    def __init__(self, hidden_dim=128, t_embed_dim=128, heads=8, radius=0.7, max_num_neighbors=5):
        super().__init__()
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors

        # Time embedding: RFF (assume GaussianFourierProjection exists)
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(t_embed_dim),
            nn.Linear(t_embed_dim, t_embed_dim)
        )

        # Coordinate projection
        self.input_proj = nn.Linear(3, hidden_dim)

        # Four GATv2 layers with residual and normalization
        self.conv1 = GATv2Conv(hidden_dim * 2, hidden_dim // heads, heads=heads, concat=True)
        self.norm1 = LayerNorm(hidden_dim)

        self.conv2 = GATv2Conv(hidden_dim * 2, hidden_dim // heads, heads=heads, concat=True)
        self.norm2 = LayerNorm(hidden_dim)

        self.conv3 = GATv2Conv(hidden_dim * 2, hidden_dim // heads, heads=heads, concat=True)
        self.norm3 = LayerNorm(hidden_dim)

        self.conv4 = GATv2Conv(hidden_dim * 2, hidden_dim // heads, heads=heads, concat=True)
        self.norm4 = LayerNorm(hidden_dim)

        self.conv5 = GATv2Conv(hidden_dim * 2, hidden_dim // heads, heads=heads, concat=True)
        self.norm5 = LayerNorm(hidden_dim)

        # time mlp
        self.mlp1 = nn.Linear(t_embed_dim, hidden_dim)
        self.mlp2 = nn.Linear(t_embed_dim, hidden_dim)
        self.mlp3 = nn.Linear(t_embed_dim, hidden_dim)
        self.mlp4 = nn.Linear(t_embed_dim, hidden_dim)
        self.mlp5 = nn.Linear(t_embed_dim, hidden_dim)

        # Output: 3D vector (score)
        self.output_proj = nn.Linear(hidden_dim, 3)

        self.act = nn.SiLU()

    def forward(self, coords, batch, t):
        t_feat = self.act(self.time_embed(t[:, None]))  # [B, t_dim]

        h = self.input_proj(coords)  # [N, hidden_dim]

        # Build edge index
        edge_index = radius_graph(
            coords,
            r=self.radius,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors
        )  # edge_index: [2, num_edges]

        # Four residual GATv2 layers
        t_per_node1 = self.mlp1(t_feat)[batch]  # [N, hidden_dim]
        x1 = self.conv1(torch.cat([h, t_per_node1], dim=-1), edge_index)
        h = self.act(self.norm1(x1, batch) + h)

        t_per_node2 = self.mlp2(t_feat)[batch]
        x2 = self.conv2(torch.cat([h, t_per_node2], dim=-1), edge_index)
        h = self.act(self.norm2(x2, batch) + h)

        t_per_node3 = self.mlp3(t_feat)[batch]
        x3 = self.conv3(torch.cat([h, t_per_node3], dim=-1), edge_index)
        h = self.act(self.norm3(x3, batch) + h)

        t_per_node4 = self.mlp4(t_feat)[batch]
        x4 = self.conv4(torch.cat([h, t_per_node4], dim=-1), edge_index)
        h = self.act(self.norm4(x4, batch) + h)

        t_per_node5 = self.mlp5(t_feat)[batch]
        x5 = self.conv5(torch.cat([h, t_per_node5], dim=-1), edge_index)
        h = self.act(self.norm5(x5, batch) + h)

        return self.output_proj(h)  # [N, 3]


class GINEScoreModel(nn.Module):
    def __init__(self, hidden_dim=256, t_embed_dim=128, num_layers=5, max_num_neighbors=30, radius=1.5, num_basis=32):
        super().__init__()
        self.max_num_neighbors = max_num_neighbors
        self.radius = radius
        self.num_basis = num_basis

        self.t_embed = nn.Sequential(
            GaussianFourierProjection(t_embed_dim, scale=30.),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                for _ in range(num_layers)
            ]
        )
        self.convs = nn.ModuleList([GINEConv(self.mlps[l], train_eps=True, edge_dim=self.num_basis + 3) for l in range(num_layers)])
        self.norms = nn.ModuleList([GraphNorm(in_channels=hidden_dim) for _ in range(num_layers)])
        self.t_mlps = nn.ModuleList([nn.Linear(t_embed_dim, hidden_dim) for _ in range(num_layers)])

        self.input_proj = nn.Linear(3, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 3)

        self.act = nn.SiLU()

    def forward(self, coords, batch, t):
        t_feat = self.act(self.t_embed(t[:, None]))         # [B, t_dim]
        h = self.input_proj(coords)  # [N, hidden_dim]

        edge_index = radius_graph(coords, r=self.radius, batch=batch, max_num_neighbors=self.max_num_neighbors)
        edge_vec = coords[edge_index[0]] - coords[edge_index[1]]
        edge_dir = F.normalize(edge_vec, dim=-1)
        edge_length = edge_vec.norm(dim=-1)

        # Positional encoding using edge length (already exists)
        edge_scalar = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.radius,
            number=self.num_basis,
            basis='gaussian',
            cutoff=True
        ) # [num_edges, num_basis]

        # Combine with direction vector
        edge_attr = torch.cat([edge_scalar, edge_dir], dim=-1)  # [num_edges, num_basis + 3]

        for conv, norm, t_mlp in zip(self.convs, self.norms, self.t_mlps):
            t_per_node = t_mlp(t_feat)[batch]

            h_res = h
            h = conv(torch.cat([h, t_per_node], dim=-1), edge_index, edge_attr) # mlp input should change to hidden_dim * 2
            # h = conv(h + t_per_node, edge_index, edge_attr)
            h = self.act(norm(h, batch) + h_res)

        return self.output_proj(h)


class PNAScoreModel(nn.Module):
    def __init__(self, train_loader, hidden_dim=128, t_embed_dim=128, num_layers=5, max_num_neighbors=30, radius=1.5, num_basis=16, aggregators=None, scalers=None, deg=None):  # deg is required by PNAConv
        super().__init__()

        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.num_basis = num_basis
        if scalers is None:
            self.scalers = ['identity', 'amplification', 'attenuation']
        if aggregators is None:
            self.aggregators = ['mean', 'max', 'min', 'std']
        if deg is None:
            self.deg = self.compute_degree_histogram(train_loader)

        self.t_embed = nn.Sequential(
            GaussianFourierProjection(t_embed_dim, scale=30.),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        self.input_proj = nn.Linear(3, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 3)

        self.t_mlps = nn.ModuleList([
            nn.Linear(t_embed_dim, hidden_dim) for _ in range(num_layers)
        ])

        self.mlps = nn.ModuleList([
            MLP([hidden_dim, hidden_dim], act='silu') for _ in range(num_layers)
        ])

        self.convs = nn.ModuleList([
            PNAConv(in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=self.aggregators,
                scalers=self.scalers,
                deg=self.deg,
                edge_dim=num_basis,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.act = nn.SiLU()

    def compute_degree_histogram(self, loader):
        radius = self.radius
        max_nodes = self.max_num_neighbors * 2
        deg = torch.zeros(max_nodes, dtype=torch.long)

        print("Computing degree histogram for PNA score model...")
        for batch in loader:
            # Flatten: from List[Tensor (L_i, 3)] → Tensor [N, 3]
            coords = torch.cat(batch, dim=0)
            batch_idx = [torch.full((x.shape[0],), i, dtype=torch.long) for i, x in enumerate(batch)]
            batch_tensor = torch.cat(batch_idx, dim=0)

            # Graph: only edge_index needed
            edge_index = radius_graph(coords, r=radius, batch=batch_tensor)

            # Degree of destination nodes
            deg_batch = degree(edge_index[1], num_nodes=coords.size(0), dtype=torch.long)
            deg += torch.bincount(deg_batch, minlength=deg.numel())
        print("Done.")

        return deg

    def forward(self, coords, batch, t):
        t_feat = self.act(self.t_embed(t[:, None]))  # [B, t_embed_dim]
        h = self.input_proj(coords)  # [N, hidden_dim]

        edge_index = radius_graph(coords, r=self.radius, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        edge_vec = coords[edge_index[0]] - coords[edge_index[1]]
        edge_length = edge_vec.norm(dim=-1)

        edge_attr = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.radius,
            number=self.num_basis,
            basis='gaussian',
            cutoff=True
        )

        for conv, norm, t_mlp in zip(self.convs, self.norms, self.t_mlps):
            t_per_node = t_mlp(t_feat)[batch]

            h_res = h
            h = conv(h + t_per_node, edge_index, edge_attr)
            h = self.act(norm(h, batch) + h_res)

        return self.output_proj(h)


class TransformerScoreModel(nn.Module):
    def __init__(self, hidden_dim=128, pos_embed_dim=128, t_embed_dim=128, num_layers=4, heads=8, max_num_neighbors=30, radius=5, num_basis=4):
        super().__init__()
        self.max_num_neighbors = max_num_neighbors
        self.radius = radius
        self.num_basis = num_basis
        self.heads = heads
        self.cat_embeddings = True

        # position embedding
        self.pos_embed = nn.Sequential(
            nn.Linear(1, pos_embed_dim),
            nn.SiLU(),
            nn.Linear(pos_embed_dim, pos_embed_dim)
        )

        # Time embedding
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(t_embed_dim, scale=30.),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        # Time MLPs per layer
        self.t_mlps = nn.ModuleList([
            nn.Linear(t_embed_dim, t_embed_dim) for _ in range(num_layers)
        ])

        # TransformerConv layers
        if self.cat_embeddings:
            in_channels = hidden_dim + pos_embed_dim + t_embed_dim
        else:
            in_channels = hidden_dim
        self.convs = nn.ModuleList([
            TransformerConv(
                in_channels=in_channels,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                beta=True,
                # dropout=0.2,
                edge_dim=self.num_basis + 3  # edge_attr not used for now
            )
            for _ in range(num_layers)
        ])

        # Norm layers
        self.norms = nn.ModuleList([
            BatchNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Input/output projection
        self.input_proj = nn.Linear(3, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 3)

        self.act = nn.SiLU()

    def forward(self, coords, batch, t):
        """
        Args:
            coords: [N, 3] - all coordinates in batch
            batch: [N] - batch index for each node
            t: [B] - time for each structure
        """
        # Compute per-node index within its graph
        num_graphs = batch.max().item() + 1
        pos_index = torch.empty_like(batch, dtype=torch.float)
        for i in range(num_graphs):
            mask = (batch == i)
            count = mask.sum()
            pos_index[mask] = torch.linspace(0, 1, steps=count, device=coords.device)
        pos_feat = self.pos_embed(pos_index[:, None])  # [N, pos_embed_dim]

        # Time embedding
        t_feat = self.act(self.t_embed(t[:, None]))  # [B, t_embed_dim]

        # Node features
        h = self.input_proj(coords)  # [N, hidden_dim]

        # Build radius graph to obtain edge index
        edge_index = radius_graph(coords, r=self.radius, batch=batch, max_num_neighbors=self.max_num_neighbors)

        #################################################################################
        ############## check neighborhood number within self.radius #####################
        # # edge_index[0] contains the source nodes (i.e., each node receiving a message)
        # src = edge_index[0]
        #
        # # Compute degrees: how many edges point to each node
        # num_nodes = coords.size(0)
        # node_degree = degree(src, num_nodes=num_nodes)
        #
        # # Print or analyze
        # print(f"Max: {node_degree.max():.0f}, Min: {node_degree.min():.0f}")
        #################################################################################
        #################################################################################

        # edge attributes
        edge_vec = coords[edge_index[0]] - coords[edge_index[1]]
        edge_dir = F.normalize(edge_vec, dim=-1)
        edge_length = edge_vec.norm(dim=-1)
        edge_scalar = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.radius,
            number=self.num_basis,
            basis='gaussian',
            cutoff=True
        )  # [num_edges, num_basis]
        # Combine with direction vector
        edge_attr = torch.cat([edge_scalar, edge_dir], dim=-1)  # [num_edges, num_basis + 3]

        for conv, norm, t_mlp in zip(self.convs, self.norms, self.t_mlps):
            t_per_node = t_mlp(t_feat)[batch]  # [N, t_embed_dim]

            # Residual
            h_res = h

            # Concat time + feature
            if self.cat_embeddings:
                h = conv(torch.cat([h, pos_feat, t_per_node], dim=-1), edge_index, edge_attr)
            else:
                h = conv(h + pos_feat + t_per_node, edge_index, edge_attr)

            # Normalization + residual
            # h = self.act(norm(h, batch) + h_res)
            h = self.act(norm(h) + h_res) # for batch norm

        return self.output_proj(h)


class GPSScoreModel(nn.Module):
    def __init__(self, hidden_dim=128, pos_embed_dim=128, t_embed_dim=128, num_layers=4, heads=8, max_num_neighbors=30, radius=5.0, num_basis=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.radius = radius
        self.max_num_neighbors = max_num_neighbors
        self.num_basis = num_basis
        self.heads = heads

        self.concat_proj = nn.Linear(hidden_dim + pos_embed_dim + t_embed_dim, hidden_dim)

        # Positional encoding via per-node order in each graph
        self.pos_embed = nn.Sequential(
            nn.Linear(1, pos_embed_dim),
            nn.SiLU(),
            nn.Linear(pos_embed_dim, pos_embed_dim)
        )

        # Time embedding
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(t_embed_dim, scale=30.),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        # Per-layer time projection
        self.t_mlps = nn.ModuleList([
            nn.Linear(t_embed_dim, t_embed_dim) for _ in range(num_layers)
        ])

        # Input/output projections
        self.input_proj = nn.Linear(3, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 3)

        # GPS layers with GINE as local conv
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            gine = GINEConv(nn=mlp, edge_dim=num_basis + 3)

            self.convs.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=gine,
                    heads=heads,
                    attn_type='multihead',
                    act='SiLU'
                )
            )

    def forward(self, coords, batch, t):
        # coords: [N, 3], batch: [N], t: [B]

        # Positional encoding (based on relative position in the chain)
        num_graphs = batch.max().item() + 1
        pos_index = torch.empty_like(batch, dtype=torch.float)
        for i in range(num_graphs):
            mask = batch == i
            pos_index[mask] = torch.linspace(0, 1, steps=mask.sum(), device=coords.device)
        pos_feat = self.pos_embed(pos_index[:, None])  # [N, pos_embed_dim]

        # Time embedding
        t_feat = self.t_embed(t[:, None])  # [B, t_embed_dim]

        # Input projection
        h = self.input_proj(coords)  # [N, hidden_dim]

        # Compute edge_index
        edge_index = radius_graph(coords, r=self.radius, batch=batch, max_num_neighbors=self.max_num_neighbors)

        # Edge attributes: edge length + direction
        edge_vec = coords[edge_index[0]] - coords[edge_index[1]]
        edge_dir = F.normalize(edge_vec, dim=-1)
        edge_length = edge_vec.norm(dim=-1)
        edge_scalar = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.radius,
            number=self.num_basis,
            basis='gaussian',
            cutoff=True
        )
        edge_attr = torch.cat([edge_scalar, edge_dir], dim=-1)

        for conv, t_mlp in zip(self.convs, self.t_mlps):
            t_per_node = t_mlp(t_feat)[batch]
            h_input = self.concat_proj(torch.cat([h, pos_feat, t_per_node], dim=-1))  # [N, hidden_dim]
            h = conv(h_input, edge_index, batch=batch, edge_attr=edge_attr)

        return self.output_proj(h)  # [N, 3]


class UNetScoreModel(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.channels = [64, 128, 128, 128]

        # encoding layers
        self.conv1 = nn.Conv1d(in_channels, self.channels[0], kernel_size=3, padding=2)
        self.conv2 = nn.Conv1d(self.channels[0], self.channels[1], kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(self.channels[1], self.channels[2], kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(self.channels[2], self.channels[3], kernel_size=3, stride=2, padding=1)

        self.dense1 = nn.Linear(embed_dim, self.channels[0])
        self.dense2 = nn.Linear(embed_dim, self.channels[1])
        self.dense3 = nn.Linear(embed_dim, self.channels[2])
        self.dense4 = nn.Linear(embed_dim, self.channels[3])

        self.gnorm1 = nn.GroupNorm(4, self.channels[0])
        self.gnorm2 = nn.GroupNorm(8, self.channels[1])
        self.gnorm3 = nn.GroupNorm(8, self.channels[2])
        self.gnorm4 = nn.GroupNorm(8, self.channels[3])

        # decoding layers
        self.tconv1 = nn.ConvTranspose1d(self.channels[3], self.channels[2], kernel_size=4, stride=2, padding=1)
        self.tconv2 = nn.ConvTranspose1d(self.channels[2] * 2, self.channels[1], kernel_size=4, stride=2, padding=1)
        self.tconv3 = nn.ConvTranspose1d(self.channels[1] * 2, self.channels[0], kernel_size=4, stride=2, padding=1)
        self.tconv4 = nn.Conv1d(self.channels[0] * 2, in_channels, kernel_size=3)

        self.tdense1 = nn.Linear(embed_dim, self.channels[2])
        self.tdense2 = nn.Linear(embed_dim, self.channels[1])
        self.tdense3 = nn.Linear(embed_dim, self.channels[0])

        self.tgnorm1 = nn.GroupNorm(8, self.channels[2])
        self.tgnorm2 = nn.GroupNorm(8, self.channels[1])
        self.tgnorm3 = nn.GroupNorm(4, self.channels[0])

        # embed utility layers
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x, t):
        """
        U-Net model
        :param x: vector with dimension [batch_size, channel_size, L]
        :param t: time with dimension [batch_size]
        :return: vector with dimension [batch_size, L]
        """
        # print(f'x: {x.shape}')
        embed_t = self.act(self.embed(t[:, None]))  # [B, embed_dim]

        # --- Encoding ---
        e1 = self.conv1(x) + self.dense1(embed_t)[:, :, None]
        e1 = self.act(self.gnorm1(e1))
        # print(f'e1: {e1.shape}')

        e2 = self.conv2(e1) + self.dense2(embed_t)[:, :, None]
        e2 = self.act(self.gnorm2(e2))
        # print(f'e2: {e2.shape}')

        e3 = self.conv3(e2) + self.dense3(embed_t)[:, :, None]
        e3 = self.act(self.gnorm3(e3))
        # print(f'e3: {e3.shape}')

        e4 = self.conv4(e3) + self.dense4(embed_t)[:, :, None]
        e4 = self.act(self.gnorm4(e4))
        # print(f'e4: {e4.shape}')

        # --- Decoding ---
        d1 = self.tconv1(e4) + self.tdense1(embed_t)[:, :, None]
        d1 = self.act(self.tgnorm1(d1))
        # print(f'd1: {d1.shape}')

        d2 = self.tconv2(torch.cat([d1, e3], dim=1)) + self.tdense2(embed_t)[:, :, None]
        d2 = self.act(self.tgnorm2(d2))
        # print(f'd2: {d2.shape}')

        d3 = self.tconv3(torch.cat([d2, e2], dim=1)) + self.tdense3(embed_t)[:, :, None]
        d3 = self.act(self.tgnorm3(d3))
        # print(f'd3: {d3.shape}')

        out = self.tconv4(torch.cat([d3, e1], dim=1))  # [B, 3, L]
        # print(f'out: {out.shape}')

        return out


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.):   # embed_dim must be divisible by 2
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        """
        Input: t of shape [B, 1]
        Output: [B, embed_dim]
        """
        x_proj = 2 * torch.pi * t * self.W  # [B, D/2]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def get_noise_conditioned_score(model, x_t, batch, t, sde):
    t_nodes = t[batch]  # [total_nodes]
    _, std = sde.marginal_prob(torch.zeros_like(x_t), t_nodes)

    if isinstance(model, UNetScoreModel):
        x_t, mask = pad_coords(x_t, batch)
        score = model(x_t, t=t)
        score, _ = unpad_coords(score, mask)
        return - score / std
    elif (
            isinstance(model, SE3ScoreModel) or
            isinstance(model, GATv2ScoreModel) or
            isinstance(model, GINEScoreModel) or
            isinstance(model, PNAScoreModel) or
            isinstance(model, TransformerScoreModel) or
            isinstance(model, GPSScoreModel)
    ):
        return - model(x_t, batch=batch, t=t) / std
    else:
        raise ValueError(f'Model type {type(model)} not recognized.')


def pad_coords(coords, batch, pad_len=30):
    """
    Pads variable-length protein coords to fixed-length for Conv1D.

    Args:
        coords: [total_nodes, 3]
        batch: [total_nodes] in 0 .. B-1
        pad_len: int

    Returns:
        padded_coords: [B, 3, pad_len]  # ready for Conv1D
        mask: [B, pad_len]  # True for valid entries, False for padded
    """
    B = batch.max().item() + 1
    split_coords = [coords[batch == i] for i in range(B)]

    padded = []
    masks = []

    for x in split_coords:
        L = x.size(0)
        if L >= pad_len:
            padded_x = x[:pad_len]  # [pad_len, 3]
            mask_x = torch.ones(pad_len, dtype=torch.bool, device=coords.device)
        else:
            pad_amt = pad_len - L
            padded_x = torch.cat([x, torch.zeros(pad_amt, 3, device=coords.device)], dim=0)  # [pad_len, 3]
            mask_x = torch.cat([torch.ones(L, dtype=torch.bool, device=coords.device),
                                torch.zeros(pad_amt, dtype=torch.bool, device=coords.device)])  # [pad_len]

        padded.append(padded_x.T)  # transpose to [3, pad_len] for Conv1D
        masks.append(mask_x)

    padded_coords = torch.stack(padded, dim=0)  # [B, 3, pad_len]
    mask = torch.stack(masks, dim=0)  # [B, pad_len]
    return padded_coords, mask

def unpad_coords(padded_output, mask):
    """
    Unpads Conv1D output back to [total_nodes, 3] format.

    Args:
        padded_output: [B, 3, pad_len]
        mask: [B, pad_len] (bool)

    Returns:
        unpadded_coords: [total_nodes, 3]
        batch: [total_nodes]
    """
    B, _, pad_len = padded_output.shape
    all_coords = []
    all_batches = []

    for i in range(B):
        valid_mask = mask[i]  # [pad_len]
        coords_i = padded_output[i, :, valid_mask].T  # [L_i, 3]
        all_coords.append(coords_i)
        all_batches.append(torch.full((coords_i.size(0),), i, device=padded_output.device, dtype=torch.long))

    unpadded_coords = torch.cat(all_coords, dim=0)  # [total_nodes, 3]
    batch = torch.cat(all_batches, dim=0)  # [total_nodes]
    return unpadded_coords, batch
