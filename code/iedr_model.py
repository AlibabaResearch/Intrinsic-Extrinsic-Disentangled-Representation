import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool

from feature_model import GNN
from cl_model import ContrastiveModel
from disentangle_model import disent_MutualMin

class IEDR(nn.Module):
    """
    IEDR main model
    """
    def __init__(self, args, device,  n_features, writer):
        super(IEDR, self).__init__()

        self.n_features = n_features
        hidden_layer = args.hidden_size 
        self.args = args
        self.writer = writer

        self.feature_embedding = nn.Embedding(self.n_features + 1, args.dim)
        self.feature_embedding.weight.data.normal_(1, 1)
        self.user_gnn = GNN(args.dim, hidden_layer)
        self.item_gnn = GNN(args.dim, hidden_layer)
        self.cont_gnn = GNN(args.dim, hidden_layer)

        self.node_weight = nn.Embedding(self.n_features + 1, 1)
        self.node_weight.weight.data.normal_(0.0,0.01)
        

        self.cl_model = ContrastiveModel(args, device)
        

        self.disent_user = disent_MutualMin(args, device)
        self.disent_item = disent_MutualMin(args, device)

        self.user_f = nn.Sequential(
                nn.Linear(args.dim, hidden_layer), 
                nn.ReLU(),
                nn.Linear(hidden_layer, args.out_dim), 
        ) 

        self.item_f = nn.Sequential(
                nn.Linear(args.dim, hidden_layer), 
                nn.ReLU(),
                nn.Linear(hidden_layer, args.out_dim), 
        ) 

        self.g = nn.Sequential(
                nn.Linear(args.dim*2, hidden_layer), 
                nn.ReLU(),
                nn.Linear(hidden_layer, 1), 
        ) 


        
    def forward(self, data_user, data_item, data_cont, index, train=True,
            infomin_train=False, epoch=None):

        x_user = data_user.x
        batch_user = data_user.batch
        x_item = data_item.x
        batch_item = data_item.batch
        x_cont = data_cont.x
        batch_cont = data_cont.batch
        label = data_user.y

        # handle pointwise features
        node_w_user = torch.squeeze(self.node_weight(x_user), 1)
        node_w_item = torch.squeeze(self.node_weight(x_item), 1)
        node_w_cont = torch.squeeze(self.node_weight(x_cont), 1)
        weight_user = torch.squeeze(global_add_pool(node_w_user, batch_user))
        weight_item = torch.squeeze(global_add_pool(node_w_item, batch_item))
        weight_cont = torch.squeeze(global_add_pool(node_w_cont, batch_cont))


        node_emb_user = self.feature_embedding(x_user)
        node_emb_item = self.feature_embedding(x_item)
        node_emb_cont = self.feature_embedding(x_cont)
        edge_index_user = data_user.edge_index
        edge_index_item = data_item.edge_index
        edge_index_cont = data_cont.edge_index

        node_patch_user = self.user_gnn(node_emb_user, edge_index_user)
        node_patch_item = self.item_gnn(node_emb_item, edge_index_item)
        node_patch_cont = self.cont_gnn(node_emb_cont, edge_index_cont)


        updated_graph_user = torch.squeeze(global_mean_pool(node_patch_user,
            batch_user))
        updated_graph_item = torch.squeeze(global_mean_pool(node_patch_item,
            batch_item))
        updated_graph_cont = torch.squeeze(global_mean_pool(node_patch_cont,
            batch_cont))


        if self.args.merge == 'prod':
            user_input = updated_graph_user * updated_graph_cont
            item_input = updated_graph_item * updated_graph_cont
        elif self.args.merge == 'add':
            user_input = updated_graph_user + updated_graph_cont
            item_input = updated_graph_item + updated_graph_cont
        
        user_all = self.user_f(user_input) 
        item_all = self.item_f(item_input) 

        if self.args.split:
            user_in, user_ex = torch.split(user_all, self.args.out_dim//2, dim=1) 
            item_in, item_ex = torch.split(item_all, self.args.out_dim//2, dim=1) 
            user_in = user_in.contiguous()
            user_ex = user_ex.contiguous()
            item_in = item_in.contiguous()
            item_ex = item_ex.contiguous()

            res = torch.sum((user_in + user_ex) * (item_in+item_ex), 1)
            y = res + weight_user + weight_item + weight_cont 
        else:
            res = torch.sum(user_all * item_all, 1)
            y = res + weight_user + weight_item 

        y = torch.sigmoid(y)

        user_cl_loss = 0
        item_cl_loss = 0
        user_dis_loss = 0
        item_dis_loss = 0
        infomin_loss = 0
        user_dis_pair = [0,0]
        item_dis_pair = [0,0]


        if train and self.args.split or True:
            if infomin_train:
                user_infomin_loss = self.disent_user.learning_loss(user_in, user_ex)
                item_infomin_loss = self.disent_item.learning_loss(item_in, item_ex)
                infomin_loss = user_infomin_loss + item_infomin_loss
                return infomin_loss
            

            user_dis_pair = self.disent_user(user_in, user_ex)
            item_dis_pair = self.disent_item(item_in, item_ex)


            if self.args.cl_w > 0:
                user_cl_loss = self.cl_model(updated_graph_user, updated_graph_cont,
                        user_in, self.user_f, index)
                item_cl_loss = self.cl_model(updated_graph_item, updated_graph_cont,
                        item_in, self.item_f, 100)

        return y, user_cl_loss, item_cl_loss, user_dis_pair, item_dis_pair 
    


