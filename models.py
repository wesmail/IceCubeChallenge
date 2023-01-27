# torch imports
import torch
import torch.nn.functional as F

# pyg imports
import torch_geometric


class GCN(torch.nn.Module):
    def __init__(self, in_feat=6, h_feat=8, num_targets=2):
        '''
        Graph Convolutional Network (GCN)
        The Graph Neural Network from the 
        “Semi-supervised Classification with Graph Convolutional Networks” paper, 
        using the GCNConv operator for message passing.
        '''
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = torch_geometric.nn.GCNConv(in_feat, h_feat)
        self.conv2 = torch_geometric.nn.GCNConv(h_feat, h_feat//2)
        self.lin = torch.nn.Linear(h_feat//2, num_targets)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index).leakyrelu()
        x = self.conv2(x, edge_index).leakyrelu()

        # 2. Readout layer
        x = torch_geometric.nn.global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x     
