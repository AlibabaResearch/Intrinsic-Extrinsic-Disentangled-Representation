import argparse
import torch
from train import Trainer 
import torch.nn.functional as F
from dataloader import Dataset 
from torch_geometric.loader import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='frappe_new', help='which dataset to use')
parser.add_argument('--rating_file', type=str, default='ratings_all.csv', help='the rating file of the dataset')
parser.add_argument('--sep', type=str, default=',', help="the seperator of the dataset, if is .csv file then is ','")
parser.add_argument('--dim', type=int, default=32, help='dimension of entity and relation embeddings')
parser.add_argument('--out_dim', type=int, default=64, help='dimension of entity and relation embeddings')
parser.add_argument('--hidden_size', type=int, default=128, help='dimension of MLP hidden size')
parser.add_argument('--l2_w', type=float, default=0.00001, help='weight of the l2 regularization term')
parser.add_argument('--cl_w', type=float, default=0.1, help='weight of the l2 regularization term')
parser.add_argument('--dis_w', type=float, default=0.1, help='weight of the l2 regularization term')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--temp', type=float, default=0.5, help='temprature for CL')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--n_epoch', type=int, default=200, help='the number of epochs')
parser.add_argument('--n_neg_cl', type=int, default=40, help='neg sample size of CL')
parser.add_argument('--n_neg_dis', type=int, default=5, help='neg sample size of dis')
parser.add_argument('--infomin_step', type=int, default=5, help='size of common item be counted')
parser.add_argument('--merge', type=str, default='prod', help='prod or sum')
parser.add_argument('--split', type=int, default=1, help='whether split intrinsic and extrinsic')
parser.add_argument('--random_seed', type=int, default=2019, help='size of common item be counted')
parser.add_argument('--patience', type=int, default=10, help='steps for early stop')
parser.add_argument('--vis_emb', type=int, default=0, help='Visualize embeddings')
parser.add_argument('--cuda', type=str, default='cuda:0', help='whether to use gpu')
parser.add_argument('--summary', type=str, default='', help='Summary to writer')
args = parser.parse_args()

args.summary = args.dataset + '_' + args.summary


device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
timestr = datetime.now().strftime('%Y%m%d')
run_folder = args.dataset +timestr
if not os.path.exists(f"../results"):
    os.mkdir(f"../results")

writer = SummaryWriter(f"../results/{run_folder}/{args.summary.replace(' ','_')}")

Info = f"""
dataset: {args.dataset}       
dim: {args.dim}
hidden_size: {args.hidden_size}
l2: {args.l2_w}
cl: {args.cl_w}
dis: {args.dis_w}
lr: {args.lr}
temp: {args.temp}
neg_sample_cl: {args.n_neg_cl}
neg_sample_dis: {args.n_neg_dis}
device: {device}
"""
print(Info)
writer.add_text("Hpyerparameters", Info)
writer.add_text("Summary", args.summary)



dataset = Dataset('../data/', args)

data_num = dataset.data_N()
feature_num = dataset.feature_N()
train_index, val_index = dataset.stat_info['train_test_split_index']

# split inner graphs
train_dataset = dataset[:train_index]
train_sep = len(train_dataset)//3
train_dataset_user = train_dataset[:train_sep]
train_dataset_item = train_dataset[train_sep:train_sep*2]
train_dataset_cont = train_dataset[train_sep*2:]
assert len(train_dataset_user) == len(train_dataset_item)
assert len(train_dataset_user) == len(train_dataset_cont)

val_dataset = dataset[train_index:val_index]
val_sep = len(val_dataset)//3
val_dataset_user = val_dataset[:val_sep]
val_dataset_item = val_dataset[val_sep:val_sep*2]
val_dataset_cont = val_dataset[val_sep*2:]
assert len(val_dataset_user) == len(val_dataset_item)
assert len(val_dataset_user) == len(val_dataset_cont)

test_dataset = dataset[val_index:]
test_sep = len(test_dataset)//3
test_dataset_user = test_dataset[:test_sep]
test_dataset_item = test_dataset[test_sep:test_sep*2]
test_dataset_cont = test_dataset[test_sep*2:]
assert len(test_dataset_user) == len(test_dataset_item)
assert len(test_dataset_user) == len(test_dataset_cont)


g1 = torch.Generator()
sampler1 = RandomSampler(range(len(train_dataset_user)), generator=g1) 
g2 = torch.Generator()
g2.set_state(g1.get_state())
sampler2 = RandomSampler(range(len(train_dataset_item)), generator=g2) 
g3 = torch.Generator()
g3.set_state(g1.get_state())
sampler3 = RandomSampler(range(len(train_dataset_cont)), generator=g3) 

n_workers = 2 
train_loader_u = DataLoader(train_dataset_user, batch_size=args.batch_size,
        sampler=sampler1, num_workers=n_workers)
train_loader_i = DataLoader(train_dataset_item, batch_size=args.batch_size,
        sampler=sampler2, num_workers=n_workers)
train_loader_c = DataLoader(train_dataset_cont, batch_size=args.batch_size,
        sampler=sampler3, num_workers=n_workers)

val_loader_u = DataLoader(val_dataset_user, batch_size=args.batch_size, num_workers=n_workers)
val_loader_i = DataLoader(val_dataset_item, batch_size=args.batch_size, num_workers=n_workers)
val_loader_c = DataLoader(val_dataset_cont, batch_size=args.batch_size, num_workers=n_workers)
test_loader_u = DataLoader(test_dataset_user, batch_size=args.batch_size, num_workers=n_workers)
test_loader_i = DataLoader(test_dataset_item, batch_size=args.batch_size, num_workers=n_workers)
test_loader_c = DataLoader(test_dataset_cont, batch_size=args.batch_size, num_workers=n_workers)

loader_list = [train_loader_u, train_loader_i, train_loader_c, val_loader_u,
        val_loader_i, val_loader_c, test_loader_u, test_loader_i, test_loader_c]

trainer = Trainer(args, feature_num, device, writer)

trainer.train(loader_list)
