# Contains the GNN model and its components
# The model is Y-shaped graph neural network (GNN) with a shared encoder and two branches for denoising and prediction tasks

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from beartype import beartype


class DMPNNConv(MessagePassing):
    """DMPNN convolution layer extending PyTorch Geometric's MessagePassing.
    Each edge undergoes two message passings with sum aggregation, one for each direction.
    Args:
        hidden_size (int): Size of the hidden representations for the convolution layer.
    """

    @beartype
    def __init__(self, hidden_size: int):
        super(DMPNNConv, self).__init__(
            aggr="add"
        )  # Sum aggregation function, most expressive aggregation as far as I know
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, edge_index, edge_attr):
        row, _ = edge_index
        # Since each edge is bidirectional, we do two message passings, one for each direction
        aggregated_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)
        reversed_message = torch.flip(
            edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]
        ).view(edge_attr.size(0), -1)

        return aggregated_message, self.linear(
            aggregated_message[row] - reversed_message
        )

    def message(self, edge_attr):
        return edge_attr


class GNNEncoder(nn.Module):
    """A GNN encoder using DMPNN convolutions.
    This encoder can operate in two modes: 'denoise' (using noisy features) or 'predict' (using clean features).
    It processes molecular graphs by performing message passing on edges, then aggregating to nodes,
    and finally pooling to create graph-level embeddings.
    Args:
        num_node_features (int): Number of input node features
        num_edge_features (int): Number of input edge features
        hidden_size (int): Size of hidden representations
        mode (str): Operating mode, either 'denoise' or 'predict'
        depth (int): Number of DMPNN convolution layers
        dropout (float): Dropout probability for regularization
    Returns:
        torch.Tensor: Graph-level embedding after global pooling
    Note:
        Contains a workaround for size mismatch issues that occur with batch size 1.
        This should be addressed in future versions.
    """

    @beartype
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_size: int,
        mode: str,
        depth: int,
        dropout: float,
    ):
        super().__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mode = mode

        # Encoder layers
        self.edge_init = nn.Linear(num_node_features + num_edge_features, hidden_size)
        self.convs = nn.ModuleList([DMPNNConv(hidden_size) for _ in range(depth)])
        self.edge_to_node = nn.Linear(num_node_features + hidden_size, hidden_size)
        self.pool = global_add_pool  # Not learnable

    def forward(self, data):
        """Forward pass through the DMPNN model.
        Args:
            data: Graph data object containing node features (x/x_noisy), edge indices,
                  edge attributes (edge_attr/edge_attr_noisy), and batch information.
        Returns:
            torch.Tensor: Global graph embedding after pooling node representations.
        Note:
            - Supports 'denoise' mode (uses noisy features) and 'predict' mode (uses clean features)
            - Includes workaround for size mismatch between edge and node representations
              that occurs with batch size 1
            - Uses residual connections and dropout in DMPNN convolution layers
        """
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch

        if self.mode == "denoise":
            x = data.x_noisy
            edge_attr = data.edge_attr_noisy
        elif self.mode == "predict":
            x = data.x
        else:
            raise ValueError("Invalid mode. Choose 'denoise' or 'predict'.")

        # Edge initialization
        row, _ = edge_index
        h_0 = F.relu(self.edge_init(torch.cat([x[row], edge_attr], dim=1)))
        h = h_0

        # DMPNN Conv layers
        for layer in self.convs:
            _, h = layer(edge_index, h)
            h += h_0
            h = F.dropout(F.relu(h), self.dropout, training=self.training)

        # Edge to node aggregation
        # Re-using the last layer's results for s
        s, _ = self.convs[-1](edge_index, h)

        # Due to a recurring error which I can't figure out, we add a check here
        # to ensure that the sizes of s and x match
        # This is a workaround and should be fixed in the future
        # Luckily, this issue only occurs for batches with batch size 1

        # Pad/truncate s to match x's size
        if s.shape[0] != x.shape[0]:
            # Create tensor with same length as x (regardless of connectivity)
            s_fixed = torch.zeros(x.shape[0], self.hidden_size, device=s.device)
            # Only use the connected nodes we have (first min(s.shape[0], x.shape[0]))
            min_len = min(s.shape[0], x.shape[0])
            s_fixed[:min_len] = s[:min_len]
            s = s_fixed

        q = torch.cat([x, s], dim=1)
        h = F.relu(self.edge_to_node(q))

        # Global pooling for the final node embeddings
        embedding = self.pool(h, batch)

        return embedding


class GNNDecoder(nn.Module):
    """A GNN decoder that reconstructs node and edge features from graph-level embeddings.
    This module takes graph-level embeddings and decodes them back to node and edge features
    by expanding the graph embeddings to match the original graph structure.
        hidden_size (int): Dimension of the input graph embeddings
        num_node_features (int): Number of output node features to decode
        num_edge_features (int): Number of output edge features to decode
        dropout (float): Dropout probability for regularization
    Attributes:
        node_lin (nn.Linear): Linear layer for decoding node features
        edge_lin (nn.Linear): Linear layer for decoding edge features
        dropout (float): Dropout probability
    """

    @beartype
    def __init__(
        self,
        hidden_size: int,
        num_node_features: int,
        num_edge_features: int,
        dropout: float,
    ):
        super().__init__()
        # Node decoding layer
        self.node_lin = nn.Linear(hidden_size, num_node_features)
        # Edge decoding layers
        self.edge_lin = nn.Linear(hidden_size, num_edge_features)
        self.dropout = dropout

    def forward(self, graph_embedding, batch, edge_index):
        """Forward pass to decode node and edge features from graph embeddings.
        Args:
            graph_embedding: Graph-level embeddings (batch_size, hidden_size)
            batch: Batch tensor indicating which graph each node belongs to
            edge_index: Edge connectivity in COO format (2, num_edges)
        Returns:
            tuple: (x_hat, edge_hat) - decoded node and edge features
        """
        # Decode node features
        batch_size = graph_embedding.size(0)
        node_counts = torch.bincount(batch)  # number of nodes in each graph

        # Expand each graph embedding for nodes
        expanded_nodes = []
        for g in range(batch_size):
            expanded_nodes.append(
                graph_embedding[g].unsqueeze(0).repeat(node_counts[g], 1)
            )

        # Concatenate along node dimension
        expanded_nodes = torch.cat(expanded_nodes, dim=0)  # total_nodes x hidden_size

        # Decode node features
        x_hat = F.dropout(expanded_nodes, p=self.dropout, training=self.training)
        x_hat = self.node_lin(x_hat)

        # Decode edge features
        # Map edges to their source node's graph
        edge_src = edge_index[0]  # Source nodes of edges
        edge_batch = batch[
            edge_src
        ]  # Batch indices of source nodes (which graph they belong to)

        # Expand graph embeddings for edges
        expanded_edges = graph_embedding[edge_batch]  # shape = (num_edges, hidden_size)

        # Decode edge features
        edge_hat = F.dropout(expanded_edges, p=self.dropout, training=self.training)
        edge_hat = self.edge_lin(edge_hat)

        return x_hat, edge_hat


@beartype
class GNNHead(nn.Module):
    """Initialize GNN prediction head for solubility prediction.
    Creates a simple FFN with two linear layers and dropout
    to predict solubility from graph embeddings.
    Args:
        hidden_size (int): Size of hidden layers and input embedding dimension
        dropout (float): Dropout rate for regularization
    """

    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        # Only some FFN layers which get the embedding as input
        self.ffn1 = nn.Linear(hidden_size, hidden_size)
        self.ffn2 = nn.Linear(hidden_size, 1)
        self.dropout = dropout

    def forward(self, graph_embedding):
        """Forward pass through the prediction head.
        Args:
            graph_embedding: Input graph embedding tensor
        Returns:
            Predicted solubility values with last dimension squeezed
        """
        x = F.relu(self.ffn1(graph_embedding))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.ffn2(x).squeeze(-1)


class GNN(nn.Module):
    """Main GNN model that combines encoder, decoder, and head components.
    Supports two modes:
    - 'denoise': Uses encoder-decoder architecture for denoising tasks
    - 'predict': Uses encoder-head architecture for prediction tasks
    The encoder processes input data and branches out to either the decoder or head
    depending on the current mode.
    Args:
        num_node_features (int): Number of input node features
        num_edge_features (int): Number of input edge features
        hidden_size (int): Hidden dimension size for all components
        depth (int): Number of layers in the encoder
        mode (str, optional): Operating mode, either 'denoise' or 'predict'. Defaults to 'denoise'
        dropout (float, optional): Dropout probability. Defaults to 0.1
    """

    @beartype
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_size: int,
        depth: int,
        mode: str = "denoise",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = GNNEncoder(
            num_node_features,
            num_edge_features,
            hidden_size=hidden_size,
            mode=mode,
            depth=depth,
            dropout=dropout,
        )
        self.head = GNNHead(hidden_size=hidden_size, dropout=dropout)
        self.decoder = GNNDecoder(
            hidden_size=hidden_size,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            dropout=dropout,
        )

    def set_mode(self, mode: str):
        """Set the mode for the encoder to specify whether to read noisy or noise-free data.
        Args:
            mode (str): The mode to set for the encoder.
        """
        self.encoder.mode = mode

    def get_embedding(self, data):
        """Get the graph embedding from the encoder.
        Args:
            data: Input data to be encoded.
        Returns:
            Graph embedding tensor from the encoder.
        """
        graph_embedding = self.encoder(data)
        return graph_embedding

    def forward(self, data):
        graph_embedding = self.encoder(data)

        if self.encoder.mode == "predict":
            prediction = self.head(graph_embedding)
            return prediction

        elif self.encoder.mode == "denoise":
            node_features, edge_features = self.decoder(
                graph_embedding, data.batch, data.edge_index
            )
            return node_features, edge_features

        else:
            raise ValueError("Invalid mode. Choose 'predict' or 'denoise'.")

    def encode(self, data):
        return self.encoder(data)
