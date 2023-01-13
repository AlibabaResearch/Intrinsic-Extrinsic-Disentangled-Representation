import torch
import torch.nn as nn 



class ContrastiveModel(nn.Module):
    
    def __init__(self, args, device, droprate=0.5):
        super(ContrastiveModel, self).__init__()

        self.args = args
        self.drop = nn.Dropout(droprate)
        self.device = device
        self.temp = args.temp
        self.neg_num = args.n_neg_cl
        self.sim = nn.CosineSimilarity(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.cl_loss_f = nn.CrossEntropyLoss()


    def augment(self, context, candidate=[0,1]):
        assert len(candidate) > 0

        aug_list = []

        if 0 in candidate:
            aug_list.append(self.aug0(context).unsqueeze(1))
        if 1 in candidate:
            aug_list.append(self.aug1(context).unsqueeze(1))
        if 2 in candidate:
            aug_list.append(self.aug2(context).unsqueeze(1))

        aug_tensor = torch.cat(aug_list, 1)

        weight = torch.ones(len(candidate))
        rand_index= torch.multinomial(weight, aug_list[0].size(0),
                replacement=True)

        res = aug_tensor[torch.arange(aug_tensor.size(0)), rand_index]

        return res 
    
    def aug0(self, context):
        """
        Augmentation via permutate context list
        """
        return context[torch.randperm(context.size(0))]
    def aug1(self, context):
        """
        Augmentation via dim level dropout 
        """
        return self.drop(context)

    def aug2(self, context):
        """
        Augmentation via returning no information 
        option1: noise
        option2: all 0
        currently return a noise of standard distribution
        """
        return torch.normal(0, 0.01, size=context.size()).to(self.device)

    def loss_f(self, pos, neg1, neg2):
        pos = pos.unsqueeze(1)
        neg1 = neg1.reshape(-1, self.neg_num)
        neg2 = neg2.reshape(-1, self.neg_num)
        #out_merge = torch.cat((pos, neg1, neg2), dim=1)
        out_merge = torch.cat((pos, neg1, neg2), dim=1)
        out_merge = self.softmax(out_merge) 
        target = torch.zeros(out_merge.size(0), dtype=torch.int64).to(self.device)

        loss = self.cl_loss_f(out_merge, target)
        return loss


    def forward(self, base, context, pos_out1, fn, index):

        aug_context = self.augment(context)
        #pos_input = torch.cat((base, aug_context),dim=1)
        if self.args.merge == 'prod':
            pos_input = base * aug_context
        elif self.args.merge == 'add':
            pos_input = base + aug_context

        pos_o2 = fn(pos_input)
        pos_out2, _ = torch.split(pos_o2, self.args.out_dim//2, dim=1)
        pos_out2 = pos_out2.contiguous()
        pos_sim = self.sim(pos_out1, pos_out2)
        pos_cl = torch.exp(pos_sim/self.temp)

        samp_weights = torch.ones(base.size(0))
        rand_index1= torch.multinomial(samp_weights, base.size(0)*self.neg_num,
                replacement=True)
        rand_index2= torch.multinomial(samp_weights, base.size(0)*self.neg_num,
                replacement=True)

        context_rep = torch.repeat_interleave(context, self.neg_num, dim=0)
        aug_context_rep = torch.repeat_interleave(aug_context, self.neg_num, dim=0)
        
        # different users, the same context
        if self.args.merge == 'prod':
            neg_o1 = fn(base[rand_index1] * context_rep)
        elif self.args.merge == 'add':
            neg_o1 = fn(base[rand_index1] + context_rep)
        neg_out1, _ = torch.split(neg_o1, self.args.out_dim//2, dim=1)
        neg_out1 = neg_out1.contiguous()
        # different users, the same augmented content
        if self.args.merge == 'prod':
            neg_o2 = fn(base[rand_index2] * aug_context_rep)
        if self.args.merge == 'add':
            neg_o2 = fn(base[rand_index2] + aug_context_rep)
        neg_out2, _ = torch.split(neg_o2, self.args.out_dim//2, dim=1)
        neg_out2 = neg_out2.contiguous()

        pos_out1_rep = torch.repeat_interleave(pos_out1, self.neg_num, dim=0)
        pos_out2_rep = torch.repeat_interleave(pos_out2, self.neg_num, dim=0)

        neg_sim1 = self.sim(pos_out1_rep, neg_out1)
        neg_cl1 = torch.exp(neg_sim1/self.temp)
        neg_sim2 = self.sim(pos_out2_rep, neg_out2)
        neg_cl2 = torch.exp(neg_sim2/self.temp)


        loss = self.loss_f(pos_cl, neg_cl1, neg_cl2)

        return loss



