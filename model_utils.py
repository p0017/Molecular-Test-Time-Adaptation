# Contains the GNN model and its components
# The model is Y-shaped graph neural network (GNN) with a shared encoder and two branches for denoising and prediction tasks

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool


class DMPNNConv(MessagePassing): 
    # Extending the MessagePassing class from PyG
    # Used for the convolutional layers in the encoder
    def __init__(self, hidden_size: int):
        super(DMPNNConv, self).__init__(aggr='add') # Sum aggregation function, most expressive aggregation as far as I know
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, edge_index, edge_attr):
        row, _ = edge_index
        # Since each edge is bidirectional, we do two message passings, one for each direction
        aggregated_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)
        reversed_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)

        return aggregated_message, self.linear(aggregated_message[row] - reversed_message)

    def message(self, edge_attr):
        return edge_attr
    

class GNNEncoder(nn.Module):
    def __init__(self, num_node_features: int, num_edge_features: int, hidden_size: int, mode: str, depth: int, dropout: float):
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
        edge_index, edge_attr, batch = data.edge_index, data.edge_attr, data.batch
        
        if self.mode == 'denoise':
            x = data.x_noisy
            edge_attr = data.edge_attr_noisy
        elif self.mode == 'predict':
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
    # Decoder for self-supervised denoising task
    # Decoding both node and edge features
    def __init__(self, hidden_size: int, num_node_features: int, num_edge_features: int, dropout: float):
        super().__init__()
        # Node decoding layer
        self.node_lin = nn.Linear(hidden_size, num_node_features)
        # Edge decoding layers
        self.edge_lin = nn.Linear(hidden_size, num_edge_features)
        self.dropout = dropout

    def forward(self, graph_embedding, batch, edge_index):
        # Decode node features
        batch_size = graph_embedding.size(0)
        node_counts = torch.bincount(batch)  # number of nodes in each graph

        # Expand each graph embedding for nodes
        expanded_nodes = []
        for g in range(batch_size):
            expanded_nodes.append(graph_embedding[g].unsqueeze(0).repeat(node_counts[g], 1))

        # Concatenate along node dimension
        expanded_nodes = torch.cat(expanded_nodes, dim=0)  # total_nodes x hidden_size

        # Decode node features
        x_hat = F.dropout(expanded_nodes, p=self.dropout, training=self.training)
        x_hat = self.node_lin(x_hat)
        
        # Decode edge features
        # Map edges to their source node's graph
        edge_src = edge_index[0]  # Source nodes of edges
        edge_batch = batch[edge_src]  # Batch indices of source nodes (which graph they belong to)
        
        # Expand graph embeddings for edges
        expanded_edges = graph_embedding[edge_batch]  # shape = (num_edges, hidden_size)
        
        # Decode edge features
        edge_hat = F.dropout(expanded_edges, p=self.dropout, training=self.training)
        edge_hat = self.edge_lin(edge_hat)

        return x_hat, edge_hat
    

class GNNHead(nn.Module):
    # Prediction Head for prediction solubility
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        # Only some FFN layers which get the embedding as input
        self.ffn1 = nn.Linear(hidden_size, hidden_size)
        self.ffn2 = nn.Linear(hidden_size, 1)
        self.dropout = dropout

    def forward(self, graph_embedding):
        x = F.relu(self.ffn1(graph_embedding))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.ffn2(x).squeeze(-1)
    

class GNN(nn.Module):
    # The main GNN model which brings together the encoder, decoder and head
    # It has two modes, denoise and predict
    # The encoder branches out to the decoder and head
    def __init__(self, num_node_features: int, num_edge_features: int, hidden_size: int, depth: int, mode: str='denoise', dropout: float=0.1):
        super().__init__()
        self.encoder = GNNEncoder(num_node_features, num_edge_features, hidden_size=hidden_size, mode=mode, depth=depth, dropout=dropout)
        self.head = GNNHead(hidden_size=hidden_size, dropout=dropout)
        self.decoder = GNNDecoder(hidden_size=hidden_size, num_node_features=num_node_features, num_edge_features=num_edge_features, dropout=dropout)

    def set_mode(self, mode: str):
        # Update the mode in the encoder
        # So the encoder knows if it needs to read noisy or noise-free data
        self.encoder.mode = mode

    def get_embedding(self, data):
        # Get the graph embedding from the encoder
        graph_embedding = self.encoder(data)
        return graph_embedding

    def forward(self, data):
        graph_embedding = self.encoder(data)

        if self.encoder.mode == 'predict':
            prediction = self.head(graph_embedding)
            return prediction
        
        elif self.encoder.mode == 'denoise':
            node_features, edge_features = self.decoder(graph_embedding, data.batch, data.edge_index)
            return node_features, edge_features
    
        else:
            raise ValueError("Invalid mode. Choose 'predict' or 'denoise'.")
        
    def encode(self, data):
        return self.encoder(data)