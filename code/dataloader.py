import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
import numpy as np
import pickle
import pandas as pd
import itertools
import os


class Dataset(InMemoryDataset):
    def __init__(self, root, args, transform=None, pre_transform=None):

        self.path = root
        self.dataset = args.dataset
        self.rating_file = args.rating_file
        self.sep = args.sep
        self.args = args

        super(Dataset, self).__init__(root, transform, pre_transform)
        try:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.stat_info = torch.load(self.processed_paths[1])
        except:
            self.process()
        self.data_num = self.stat_info['data_num']
        self.feature_num = self.stat_info['feature_num']

    @property
    def raw_file_names(self):
        return ['{}{}/user_dict.pkl'.format(self.path, self.dataset),
                '{}{}/item_dict.pkl'.format(self.path, self.dataset),
                '{}{}/feature_dict.pkl'.format(self.path, self.dataset),
                '{}{}/train_data.csv'.format(self.path, self.dataset),
                '{}{}/valid_data.csv'.format(self.path, self.dataset),
                '{}{}/test_data.csv'.format(self.path, self.dataset)]

    @property
    def processed_file_names(self):
        return ['{}/{}.dataset'.format(self.dataset, self.dataset),
                '{}/{}.statinfo'.format(self.dataset, self.dataset)]


    def download(self):
        # Download to `self.raw_dir`.
        pass


    def data_2_graphs(self, ratings_df, dataset='train'):
        graphs_u = []
        graphs_i = []
        graphs_c = []
        processed_graphs = 0
        num_graphs = ratings_df.shape[0]
        one_per = int(num_graphs/100)
        percent = 0.0
        for i in range(len(ratings_df)):
            if processed_graphs % one_per == 0:
                print(f"Processing [{dataset}]: {percent}%, {processed_graphs}/{num_graphs}", end="\r")
                percent += 1
            processed_graphs += 1 
            line = ratings_df.iloc[i]
            user_index = self.user_key_type(line[0])
            item_index = self.item_key_type(line[1])
            rating = int(line[2])
            context_list = [int(l)for l in line[3:]]


            if item_index not in self.item_dict or user_index not in self.user_dict:
                self.error_num += 1
                print(item_index, item_index in self.item_dict)
                print(user_index, user_index in self.user_dict)
                exit()
                continue

            user_id = self.user_dict[user_index]['name']
            item_id = self.item_dict[item_index]['title']

            user_attr_list = self.user_dict[user_index]['attribute']
            item_attr_list = self.item_dict[item_index]['attribute']

            user_list = [user_id] + user_attr_list
            item_list = [item_id] + item_attr_list

            graph_u, graph_i, graph_c = self.construct_graphs(user_list, item_list,
                    context_list, rating)

            graphs_u.append(graph_u)
            graphs_i.append(graph_i)
            graphs_c.append(graph_c)

        print()

        return graphs_u + graphs_i + graphs_c



    def read_data(self):
        self.user_dict = pickle.load(open(self.userfile, 'rb'))
        self.item_dict = pickle.load(open(self.itemfile, 'rb'))
        self.user_key_type = type(list(self.user_dict.keys())[0])
        self.item_key_type = type(list(self.item_dict.keys())[0])
        feature_dict = pickle.load(open(self.featurefile, 'rb'))

        self.error_num = 0

        train_df = pd.read_csv(self.trainfile, sep=self.sep)
        valid_df = pd.read_csv(self.validfile, sep=self.sep)
        test_df = pd.read_csv(self.testfile, sep=self.sep)
                
        print('(Only run at the first time training the dataset)')
        train_graphs = self.data_2_graphs(train_df, dataset='train')
        valid_graphs = self.data_2_graphs(valid_df, dataset='valid')
        test_graphs = self.data_2_graphs(test_df, dataset='test')

        graphs = train_graphs + valid_graphs + test_graphs 

        stat_info = {}
        stat_info['data_num'] = len(graphs)
        stat_info['feature_num'] = len(feature_dict)
        stat_info['train_test_split_index'] = [len(train_graphs), len(train_graphs) + len(valid_graphs)]

        print('error number of data:', self.error_num)
        return graphs, stat_info


    def construct_graphs(self, user_list, item_list, cont_list, rating):
        u_n = len(user_list)   # user node number
        i_n = len(item_list)   # item node number
        c_n = len(cont_list)

        # construct full inner edge
        edge_index_user = np.array(list(itertools.product(range(u_n), range(u_n))))
        edge_index_user = np.transpose(edge_index_user)

        edge_index_item = np.array(list(itertools.product(range(i_n), range(i_n))))
        edge_index_item = np.transpose(edge_index_item)

        edge_index_cont = np.array(list(itertools.product(range(c_n), range(c_n))))
        edge_index_cont = np.transpose(edge_index_cont)

        #construct graph
        edge_index_user = torch.LongTensor(edge_index_user)
        edge_index_item = torch.LongTensor(edge_index_item)
        edge_index_cont = torch.LongTensor(edge_index_cont)

        graph_user = self.construct_graph(user_list, edge_index_user, rating)
        graph_item = self.construct_graph(item_list, edge_index_item, rating)
        graph_cont = self.construct_graph(cont_list, edge_index_cont, rating)

        return graph_user, graph_item, graph_cont


    def construct_graph(self, node_list, edge_index, rating):
        x = torch.LongTensor(node_list).unsqueeze(1)
        rating = torch.FloatTensor([rating])
        data = Data(x=x, edge_index=edge_index,  y=rating)
        return data 

    def process(self):
        if not os.path.exists(f"{self.path}processed"):
            os.mkdir(f"{self.path}processed")

        self.userfile  = self.raw_file_names[0]
        self.itemfile  = self.raw_file_names[1]
        self.featurefile = self.raw_file_names[2]
        self.trainfile = self.raw_file_names[3]
        self.validfile = self.raw_file_names[4]
        self.testfile = self.raw_file_names[5]
        graphs, self.stat_info = self.read_data()
        #check whether foler path exist
        if not os.path.exists(f"{self.path}processed/{self.dataset}"):
            os.mkdir(f"{self.path}processed/{self.dataset}")

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

        torch.save(self.stat_info, self.processed_paths[1])

        #Unknown issue: crash at the first time generating the data 
        print("Please the code again")
        exit()

    def feature_N(self):
        return self.feature_num

    def data_N(self):
        return self.data_num


