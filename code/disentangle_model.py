import torch
import torch.nn as nn 


class disent_disCor(nn.Module):
    """
    the distance correaltion based disentanglement method
    the correlation will be regardedd as loss to reduce
    """
    def __init__(self, args, device):
        super(disent_disCor, self).__init__()
        self.args = args,
        self.device = device

    def dcov(self, v1, v2, mean1, mean2):
        d1 = torch.cdist(v1, mean1)
        d2 = torch.cdist(v2, mean2)

        return torch.mean(d1*d2)



    def forward(self, v_in, v_ex):
        in_mean = torch.mean(v_in,
                dim=0).unsqueeze(0).repeat((v_in.size(0),1)).unsqueeze(1)
        ex_mean = torch.mean(v_ex,
                dim=0).unsqueeze(0).repeat((v_ex.size(0),1)).unsqueeze(1)

        v_in = v_in.unsqueeze(1)
        v_ex = v_ex.unsqueeze(1)

        cov_ie = self.dcov(v_in, v_ex, in_mean, ex_mean) 
        cov_ii = self.dcov(v_in, v_in, in_mean, in_mean)
        cov_ee = self.dcov(v_ex, v_ex, ex_mean, ex_mean)
        assert cov_ie > 0
        assert cov_ii > 0
        assert cov_ee > 0
        res = cov_ie/torch.sqrt(cov_ii*cov_ee)

        return res         



class disent_MutualMin(nn.Module):
    """
    Leveraging the mutual information minimization methods based 
    on the paper about contrastive log-ratio upper bound method.
    """
    def __init__(self, args, device):
        super(disent_MutualMin, self).__init__()
        self.args = args,
        self.device = device

        self.f_estimate = nn.Sequential(
                nn.Linear(args.dim, args.hidden_size), 
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.dim), 
        ) 
        self.f_estimate_rev = nn.Sequential(
                nn.Linear(args.dim, args.hidden_size), 
                nn.ReLU(),
                nn.Linear(args.hidden_size, args.dim), 
        )
        self.loss = nn.MSELoss()

        self.neg_num = args.n_neg_dis 

    def forward(self, v_in, v_ex):
        pos_ex_est = self.f_estimate(v_in)

        samp_weights = torch.ones(v_in.size(0))
        rand_index = torch.multinomial(samp_weights, v_in.size(0)*self.neg_num,
                replacement=True)

        neg_ex = torch.repeat_interleave(v_ex, self.neg_num, dim=0) 
        neg_ex_est = pos_ex_est[rand_index]

        pos_loss = self.loss(pos_ex_est, v_ex)
        neg_loss = self.loss(neg_ex_est, neg_ex)

        # reverse procedure
        pos_in_est = self.f_estimate_rev(v_ex)
        rand_index = torch.multinomial(samp_weights, v_ex.size(0)*self.neg_num,
                replacement=True)
        neg_in = torch.repeat_interleave(v_in, self.neg_num, dim=0)
        neg_in_est = pos_in_est[rand_index]

        pos_loss_rev = self.loss(pos_in_est, v_in)
        neg_loss_rev = self.loss(neg_in_est, neg_in)

        return neg_loss+neg_loss_rev, pos_loss+pos_loss_rev

    
    def learning_loss(self, v_in, v_ex):
        pos_ex_est = self.f_estimate(v_in)
        pos_loss = self.loss(pos_ex_est, v_ex)

        pos_in_est = self.f_estimate_rev(v_ex)
        pos_loss_rev = self.loss(pos_in_est, v_in)
        return pos_loss+pos_loss_rev



