import torch.nn as nn 
from torch_geometric.nn.conv import MessagePassing


class GNN(MessagePassing):
    def __init__(self, dim, hidden_layer):
        super(GNN, self).__init__(aggr='mean')

        #construct pairwise modeling network
        self.lin1 = nn.Linear(dim, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, dim]
        # edge_index has shape [2, E]
        #try:
        #    return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        #except:
        x = x.squeeze()
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        # x_i has shape [E, dim]
        # x_j has shape [E, dim]

        # pairwise analysis
        pairwise_analysis = x_i * x_j
        pairwise_analysis = self.lin1(pairwise_analysis)
        pairwise_analysis = self.act(pairwise_analysis)
        pairwise_analysis = self.lin2(pairwise_analysis)
        pairwise_analysis = self.drop(pairwise_analysis)

        if edge_weight != None:
            interaction_analysis = pairwise_analysis * edge_weight.view(-1,1)
        else:
            interaction_analysis = pairwise_analysis

        return interaction_analysis

    def update(self, aggr_out):
        # aggr_out has shape [N, dim]
        return aggr_out


