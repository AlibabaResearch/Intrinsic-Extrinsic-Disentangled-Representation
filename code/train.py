import torch
from torch import nn
from iedr_model import IEDR 
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, ndcg_score
import numpy as np
import pandas as pd 
import time
import os


class Trainer():
    def __init__(self, args, feature_num,  device, writer):
        self.device = device
        self.training_loss = []
        self.model = IEDR(args, device, feature_num, writer).to(device)
        self.args = args
        self.writer = writer

    def train(self, loader_l):
        start_time = time.time()
        infomin_para = [param for name, param in filter(lambda p: p[1].requires_grad,
            self.model.named_parameters()) if 'disent' in name]
        model_para = [param for name, param in filter(lambda p: p[1].requires_grad,
            self.model.named_parameters()) if 'disent' not in name]

        optimizer = torch.optim.Adam(model_para,
                weight_decay=self.args.l2_w, lr=self.args.lr)
        optimizer_infomin = torch.optim.Adam(infomin_para,
                weight_decay=self.args.l2_w, lr=self.args.lr)
             
        loss_f = nn.BCELoss()

        print([(name, param.size()) for name, param in filter(lambda p: p[1].requires_grad,
            self.model.named_parameters())])

        
        best_ndcg = 0
        best_t = 0
        cnt_wait = 0

        for epoch in range(self.args.n_epoch):
            base_loss = 0
            cl_loss_l = 0
            dis_loss_l = 0
            dis_loss_ln = 0
            index = 1000000 
            for data_user, data_item, data_cont in zip(loader_l[0],
                    loader_l[1], loader_l[2]):
                self.model.train()
                data_user = data_user.to(self.device)
                data_item = data_item.to(self.device)
                data_cont = data_cont.to(self.device)

                pred, user_cl_loss, item_cl_loss, user_dis_pair, item_dis_pair\
                = self.model(data_user, data_item, data_cont, index)
                index -= 1
                
                baseloss = loss_f(pred, data_item.y)
                cl_loss = user_cl_loss + item_cl_loss
                
                dis_loss = user_dis_pair[0] + item_dis_pair[0]\
                    - user_dis_pair[1] - item_dis_pair[1]

                loss = baseloss + self.args.cl_w*cl_loss + self.args.dis_w*dis_loss
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                base_loss += baseloss.item()
                cl_loss_l += cl_loss
                dis_loss_l += user_dis_pair[1] + item_dis_pair[1] 
                dis_loss_ln += user_dis_pair[0] + item_dis_pair[0] 

                for i in range(self.args.infomin_step):
                    infomin_loss = self.model(data_user, data_item, data_cont,
                            index, infomin_train=True)
                    optimizer_infomin.zero_grad()
                    infomin_loss.backward()
                    optimizer_infomin.step()
                

            base_loss = base_loss/len(loader_l[0])
            cl_loss_l = cl_loss_l/len(loader_l[0])
            dis_loss_l = dis_loss_l/len(loader_l[0])
            dis_loss_ln = dis_loss_ln/len(loader_l[0])
            print(f"Epoch {epoch}, baseloss: {base_loss:.4f},", end=' ') 
            #print(f"cl_loss:{cl_loss_l:.4f}, dis_loss: {dis_loss_l:.4f}, {dis_loss_ln:.4f} ", end='' )

            val_results = self.evaluate(loader_l[3], loader_l[4], loader_l[5],
                    self.device, epoch=epoch)
            print(f"val_AUC: {val_results[0]:.5f}, ", end=' ')
            print(f"ndcg@5: {val_results[1][0]:.5f}, ndcg@10: {val_results[1][1]:.5f} ", end= ' ')
            print(f"hr@5: {val_results[2][0]:.5f}, hr@10: {val_results[2][1]:.5f}")

            # Write results to Tensorboard
            self.writer.add_scalar("results/Val AUC", val_results[0], epoch)
            self.writer.add_scalar("results/Val NDCG@5", val_results[1][0], epoch)
            self.writer.add_scalar("results/Val NDCG@10", val_results[1][1], epoch)
            self.writer.add_scalar("results/Val HR@5", val_results[2][0], epoch)
            self.writer.add_scalar("results/Val HR@10", val_results[2][1], epoch)


            self.writer.add_scalar("Loss/baseloss", base_loss, epoch)
            self.writer.add_scalar("Loss/CL_loss", cl_loss_l, epoch)
            self.writer.add_scalar("Loss/Dis_loss pos", dis_loss_l, epoch)
            self.writer.add_scalar("Loss/Dis_loss neg", dis_loss_ln, epoch)
            self.writer.add_scalar("Loss/MI", dis_loss_ln-dis_loss_l, epoch)
            
            if val_results[1][1] > best_ndcg:
                best_ndcg = val_results[1][1]
                best_t = epoch
                cnt_wait = 0
                if not os.path.exists(f"../results/{self.args.dataset}"):
                    os.mkdir(f"../results/{self.args.dataset}")
                    
                torch.save(self.model.state_dict(),
                        f"../results/{self.args.dataset}/{self.args.summary}_{start_time}.pkl")
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                print("Early Stopping")
                break

               

        print(f"Loading {best_t}th epoch")
        self.model.load_state_dict(torch.load(f"../results/{self.args.dataset}/{self.args.summary}_{start_time}.pkl"))
        print("Test Results:")
        test_res = self.evaluate(loader_l[6], loader_l[7], loader_l[8], self.device)
        print(test_res)
        hyperpara = vars(self.args)
        eval_res = {"t_AUC": test_res[0], 
                    "t_NDCG@5": test_res[1][0],
                    "t_NDCG@10": test_res[1][1],
                    "t_HR@5": test_res[2][0],
                    "t_HR@10": test_res[2][1],
                    }
        self.writer.add_hparams(hyperpara, eval_res)


    def evaluate(self, loader_u, loader_i, loader_c, device, epoch=None):
        self.model.eval()

        predictions = []
        labels = []
        user_ids = []
        ii = 1000
        with torch.no_grad():
            for data_user, data_item, data_cont in zip(loader_u, loader_i, loader_c):
                data_user = data_user.to(self.device)
                data_item = data_item.to(self.device)
                data_cont = data_cont.to(self.device)

                _, user_id_index = np.unique(data_user.batch.detach().cpu().numpy(), return_index=True)
                user_id = data_user.x.detach().cpu().numpy()[user_id_index]
                user_ids.append(user_id)

                pred, _, _, _, _  = self.model(data_user, data_item, data_cont,
                        ii, train=False, epoch=epoch)
                pred = pred.squeeze().detach().cpu().numpy().astype('float64')
                
                label = data_item.y.detach().cpu().numpy()
                if pred.size == 1:
                    pred = np.expand_dims(pred, axis=0)
                predictions.append(pred)
                labels.append(label)
                ii -= 1

        predictions = np.concatenate(predictions, 0)
        labels = np.concatenate(labels, 0)
        user_ids = np.concatenate(user_ids, 0)

        ndcg_list, hr_list = self.cal_rank(predictions, labels, user_ids, [5,10])

        auc = roc_auc_score(labels, predictions)

        return auc, ndcg_list, hr_list


    def cal_rank(self, predicts, labels, user_ids, k_list):
        d = {'user': np.squeeze(user_ids), 'predict':np.squeeze(predicts), 'label':np.squeeze(labels)}
        df = pd.DataFrame(d)
        user_unique = df.user.unique()


        ndcg_list = [[] for _ in range(len(k_list))]
        hr_list = [[] for _ in range(len(k_list))]
        for user_id in user_unique:
            user_srow = df[df['user'] == user_id]

            # cal ndcg
            upred = user_srow['predict'].tolist()
            ulabel = user_srow['label'].tolist()
            for i in range(len(k_list)):
                ndcg_list[i].append(ndcg_score([ulabel], [upred], k=k_list[i])) 

            # cal hr
            user_sdf = user_srow.sort_values(by=['predict'], ascending=False)
            total_rel = len(user_sdf)

            for i in range(len(k_list)):
                intersect_at_k = user_sdf['label'][0:k_list[i]].sum()/len(user_sdf)*100
                hr_list[i].append(float(intersect_at_k))

        ndcg_res = np.mean(np.array(ndcg_list), axis=1)
        hr_res = np.mean(np.array(hr_list), axis=1)
 

        return ndcg_res, hr_res 

