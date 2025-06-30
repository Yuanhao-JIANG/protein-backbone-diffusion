import torch
import torch.nn as nn
from e3nn.o3 import Irreps, spherical_harmonics
from e3nn.math import soft_one_hot_linspace
from e3nn.nn.models.gate_points_2101 import Convolution
from torch_cluster import radius_graph


class SE3ScoreModel(nn.Module):
    def __init__(self, hidden_dim=32, num_neighbors=32, radius=5.0):
        super().__init__()

        self.radius = radius
        self.num_neighbors = num_neighbors

        # Define irreps clearly
        irreps_node_input = Irreps("0e")  # Scalar features (initially no direction)
        irreps_node_attr = Irreps("0e")  # Node attributes (scalar)
        irreps_edge_attr = Irreps.spherical_harmonics(lmax=1)  # "1o"
        irreps_hidden = Irreps(f"{hidden_dim}x0e + {hidden_dim}x1o")
        irreps_output = Irreps("1o")  # Vectorial output (3-dimensional)

        self.conv1 = Convolution(
            irreps_in=irreps_node_input,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_out=irreps_hidden,
            number_of_basis=10,
            radial_layers=2,
            radial_neurons=hidden_dim,
            num_neighbors=num_neighbors
        )

        self.conv2 = Convolution(
            irreps_in=irreps_hidden,
            irreps_node_attr=irreps_node_attr,
            irreps_edge_attr=irreps_edge_attr,
            irreps_out=irreps_output,
            number_of_basis=10,
            radial_layers=2,
            radial_neurons=hidden_dim,
            num_neighbors=num_neighbors
        )

        self.act = nn.SiLU()

    def forward(self, coords, batch, t=None):
        device = coords.device
        N = coords.shape[0]

        # Node features
        node_feats = torch.ones(N, 1, device=device)

        # Edges and edge attributes
        edge_src, edge_dst = radius_graph(
            coords, r=self.radius, batch=batch, max_num_neighbors=self.num_neighbors
        )
        edge_vec = coords[edge_dst] - coords[edge_src]

        edge_sh = spherical_harmonics(
            l=[0,1], x=edge_vec, normalize=True, normalization='component'
        )  # [num_edges, 3]

        edge_length = edge_vec.norm(dim=1)  # [num_edges]

        # Radial embedding (fix)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.radius,
            number=10,  # match your convolution's number_of_basis parameter
            basis='gaussian',
            cutoff=True
        )  # [num_edges, number_of_basis]

        # Node attributes (timestep embedding or scalar)
        if t is not None:
            node_attr = t[batch].unsqueeze(-1)
        else:
            node_attr = torch.ones(N, 1, device=device)

        # Convolution layers
        node_hidden = self.conv1(
            node_input=node_feats,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_length_embedded=edge_length_embedded
        )
        node_hidden = self.act(node_hidden)

        node_output = self.conv2(
            node_input=node_hidden,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_length_embedded=edge_length_embedded
        )

        return node_output
