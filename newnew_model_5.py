# -- coding: utf-8 --
# @Time: 2022-04-10 15:18
# @Author: WangCx
# @File: newnew_model
# @Project: HypergraphNN_test
import torch
from torch.nn.init import xavier_normal_

import torch.nn.functional as F
import math


class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)


class HyperNet(BaseClass):
    def __init__(self, dataset, emb_dim, hidden_drop):
        super(HyperNet, self).__init__()
        self.emb_dim = emb_dim
        self.dataset = dataset
        self.E = torch.nn.Embedding(dataset.num_ent, emb_dim, padding_idx=0)
        self.R = torch.nn.Embedding(dataset.num_rel, emb_dim, padding_idx=0)
        self.W0 = torch.nn.Parameter(torch.empty(size=(3*emb_dim, 300)))
        self.W1 = torch.nn.Parameter(torch.empty(size=(2*emb_dim, 300)))
        self.W2 = torch.nn.Parameter(torch.empty(size=(emb_dim, 300)))
        self.W3 = torch.nn.Parameter(torch.empty(size=(emb_dim, 300)))
        self.a = torch.nn.Parameter(torch.empty(size=(300, 1)))
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
        xavier_normal_(self.W0.data)
        xavier_normal_(self.a.data)
        xavier_normal_(self.W1.data)
        xavier_normal_(self.W2.data)
        xavier_normal_(self.W3.data)
        self.hidden_drop_rate = hidden_drop
        self.hidden_drop = torch.nn.Dropout(self.hidden_drop_rate)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)


        self.in_channels = 1
        self.out_channels = 6
        self.filt_h = 1
        self.filt_w = 1
        self.stride = 2

        self.max_arity = self.dataset.arity_lst[-1]


        self.bn0 = torch.nn.BatchNorm2d(self.in_channels)
        self.inp_drop = torch.nn.Dropout(0.2)

        fc_length = (1-self.filt_h+1)*math.floor((emb_dim-self.filt_w)/self.stride + 1)*self.out_channels

        self.bn2 = torch.nn.BatchNorm1d(fc_length)
        # Projection network
        self.fc = torch.nn.Linear(fc_length, emb_dim)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # size of the convolution filters outputted by the hypernetwork
        fc1_length = self.in_channels*self.out_channels*self.filt_h*self.filt_w
        # Hypernetwork
        self.fc1 = torch.nn.Linear(emb_dim + self.max_arity + 1, fc1_length) # (306, 5)
        self.fc2 = torch.nn.Linear(self.max_arity + 1, fc1_length) # (6, 5)


    def er_pos_emb(self, r_emb, e_emb):
        return torch.mm(torch.cat((r_emb, e_emb), dim=1), self.W1)


    def convolve(self, r, ei, pos):

        e = ei.view(-1, 1, 1, self.E.weight.size(1))
        x = e
        x = self.inp_drop(x)
        one_hot_target = (pos == torch.arange(self.max_arity + 1).reshape(self.max_arity + 1)).float().to(self.device)
        poses = one_hot_target.repeat(r.shape[0]).view(-1, self.max_arity + 1)
        one_hot_target.requires_grad = False
        poses.requires_grad = False
        k = self.fc2(poses)
        k = k.view(-1, self.in_channels, self.out_channels, self.filt_h, self.filt_w)
        k = k.view(e.size(0)*self.in_channels*self.out_channels, 1, self.filt_h, self.filt_w)
        x = x.permute(1, 0, 2, 3)
        x = F.conv2d(x, k, stride=self.stride, groups=e.size(0))
        x = x.view(e.size(0), 1, self.out_channels, 1-self.filt_h+1, -1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(e.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, batch, ms, bs):
        r_idx = batch[:, 0]
        r = self.R(r_idx)

        if batch.shape[1] == 3:

            e1 = self.E(batch[:, 1])
            e2 = self.E(batch[:, 2])
            e1_pos = self.convolve(r, e1, 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
            e2_pos = self.convolve(r, e2, 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)

            e11 = torch.mm(torch.cat((e1_pos, e1_pos, r), dim=1), self.W0)
            e12 = torch.mm(torch.cat((e1_pos, e2_pos, r), dim=1), self.W0)
            e21 = torch.mm(torch.cat((e2_pos, e1_pos, r), dim=1), self.W0)
            e22 = torch.mm(torch.cat((e2_pos, e2_pos, r), dim=1), self.W0)

            e1_all_att = torch.zeros(e1.shape[0], 1)
            e2_all_att = torch.zeros(e2.shape[0], 1)

            i1 = 0
            i2 = 0
            for e1_ in batch[:, 1]:
                e1_int = e1_.item()
                att_sum_1 = 0
                for hyperedge in self.dataset.inc[e1_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1: ]:
                        ei = self.E(e1_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_1 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e1_all_att[i] = att_sum_1
                i += 1

            for e2_ in batch[:, 2]:
                e2_int = e2_.item()
                att_sum_2 = 0
                for hyperedge in self.dataset.inc[e2_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1: ]:
                        ei = self.E(e2_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_2 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e2_all_att[i2] = att_sum_2
                i2 += 1




            e1_e1_att = torch.exp(self.leakyrelu(torch.mm(e11, self.a))) / e1_all_att
            e1_e2_att = torch.exp(self.leakyrelu(torch.mm(e12, self.a))) / e1_all_att
            e2_e1_att = torch.exp(self.leakyrelu(torch.mm(e21, self.a))) / e2_all_att
            e2_e2_att = torch.exp(self.leakyrelu(torch.mm(e22, self.a))) / e2_all_att


            new_e1 = torch.mm(e1, self.W2) + torch.tanh(e11*e1_e1_att + e12*e1_e2_att)
            new_e2 = torch.mm(e2, self.W2) + torch.tanh(e21*e2_e1_att + e22*e2_e2_att)

            re1 = self.er_pos_emb(r, e1_pos)
            re2 = self.er_pos_emb(r, e2_pos)
            re1_att = torch.exp(torch.cosine_similarity(r, e1, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)))
            re2_att = torch.exp(torch.cosine_similarity(r, e2, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)))

            r = torch.mm(r, self.W3) + torch.tanh(re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1))


            e = new_e1 * new_e2 * r

        elif batch.shape[1] == 4:

            e1 = self.E(batch[:, 1])
            e2 = self.E(batch[:, 2])
            e3 = self.E(batch[:, 3])
            e1_pos = self.convolve(r, e1, 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
            e2_pos = self.convolve(r, e2, 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
            e3_pos = self.convolve(r, e3, 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)

            e11 = torch.mm(torch.cat((e1_pos, e1_pos, r), dim=1), self.W0)
            e12 = torch.mm(torch.cat((e1_pos, e2_pos, r), dim=1), self.W0)
            e13 = torch.mm(torch.cat((e1_pos, e3_pos, r), dim=1), self.W0)
            e21 = torch.mm(torch.cat((e2_pos, e1_pos, r), dim=1), self.W0)
            e22 = torch.mm(torch.cat((e2_pos, e2_pos, r), dim=1), self.W0)
            e23 = torch.mm(torch.cat((e2_pos, e3_pos, r), dim=1), self.W0)
            e31 = torch.mm(torch.cat((e3_pos, e1_pos, r), dim=1), self.W0)
            e32 = torch.mm(torch.cat((e3_pos, e2_pos, r), dim=1), self.W0)
            e33 = torch.mm(torch.cat((e3_pos, e3_pos, r), dim=1), self.W0)


            re1 = self.er_pos_emb(r, e1_pos)
            re2 = self.er_pos_emb(r, e2_pos)
            re3 = self.er_pos_emb(r, e3_pos)
            re1_att = torch.exp(torch.cosine_similarity(r, e1, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)))
            re2_att = torch.exp(torch.cosine_similarity(r, e2, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)))
            re3_att = torch.exp(torch.cosine_similarity(r, e3, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)))

            r = torch.mm(r, self.W3) + torch.tanh(re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1))

            # r = re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1)
            new_e1 = torch.mm(e1, self.W2) + torch.tanh(e11*e1_e1_att + e12*e1_e2_att + e13*e1_e3_att)
            new_e2 = torch.mm(e2, self.W2) + torch.tanh(e21*e2_e1_att + e22*e2_e2_att + e23*e2_e3_att)
            new_e3 = torch.mm(e3, self.W2) + torch.tanh(e31*e3_e1_att + e32*e3_e2_att + e33*e3_e3_att)


            e = new_e1 * new_e2 * new_e3 * r

        elif batch.shape[1] == 5:
            e1 = self.E(batch[:, 1])
            e2 = self.E(batch[:, 2])
            e3 = self.E(batch[:, 3])
            e4 = self.E(batch[:, 4])
            e1_pos = self.convolve(r, e1, 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
            e2_pos = self.convolve(r, e2, 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
            e3_pos = self.convolve(r, e3, 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
            e4_pos = self.convolve(r, e4, 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)


            e11 = torch.mm(torch.cat((e1_pos, e1_pos, r), dim=1), self.W0)
            e12 = torch.mm(torch.cat((e1_pos, e2_pos, r), dim=1), self.W0)
            e13 = torch.mm(torch.cat((e1_pos, e3_pos, r), dim=1), self.W0)
            e14 = torch.mm(torch.cat((e1_pos, e4_pos, r), dim=1), self.W0)
            e21 = torch.mm(torch.cat((e2_pos, e1_pos, r), dim=1), self.W0)
            e22 = torch.mm(torch.cat((e2_pos, e2_pos, r), dim=1), self.W0)
            e23 = torch.mm(torch.cat((e2_pos, e3_pos, r), dim=1), self.W0)
            e24 = torch.mm(torch.cat((e2_pos, e4_pos, r), dim=1), self.W0)
            e31 = torch.mm(torch.cat((e3_pos, e1_pos, r), dim=1), self.W0)
            e32 = torch.mm(torch.cat((e3_pos, e2_pos, r), dim=1), self.W0)
            e33 = torch.mm(torch.cat((e3_pos, e3_pos, r), dim=1), self.W0)
            e34 = torch.mm(torch.cat((e3_pos, e4_pos, r), dim=1), self.W0)
            e41 = torch.mm(torch.cat((e4_pos, e1_pos, r), dim=1), self.W0)
            e42 = torch.mm(torch.cat((e4_pos, e2_pos, r), dim=1), self.W0)
            e43 = torch.mm(torch.cat((e4_pos, e3_pos, r), dim=1), self.W0)
            e44 = torch.mm(torch.cat((e4_pos, e4_pos, r), dim=1), self.W0)

            i1 = 0
            i2 = 0
            i3 = 0
            i4 = 0
            e1_all_att = torch.zeros(e1.shape[0], 1)
            e2_all_att = torch.zeros(e2.shape[0], 1)
            e3_all_att = torch.zeros(e3.shape[0], 1)
            e4_all_att = torch.zeros(e4.shape[0], 1)

            for e1_ in batch[:, 1]:
                e1_int = e1_.item()
                att_sum_1 = 0
                for hyperedge in self.dataset.inc[e1_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1:]:
                        ei = self.E(e1_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_1 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e1_all_att[i] = att_sum_1
                i += 1

            for e2_ in batch[:, 2]:
                e2_int = e2_.item()
                att_sum_2 = 0
                for hyperedge in self.dataset.inc[e2_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1:]:
                        ei = self.E(e2_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_2 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e2_all_att[i2] = att_sum_2
                i2 += 1

            for e3_ in batch[:, 3]:
                e3_int = e3_.item()
                att_sum_3 = 0
                for hyperedge in self.dataset.inc[e3_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1:]:
                        ei = self.E(e3_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_3 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e3_all_att[i3] = att_sum_3
                i3 += 1

            for e4_ in batch[:, 4]:
                e4_int = e4_.item()
                att_sum_4 = 0
                for hyperedge in self.dataset.inc[e4_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1:]:
                        ei = self.E(e4_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_4 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e4_all_att[i4] = att_sum_4
                i4 += 1

            e1_e1_att = torch.exp(self.leakyrelu(torch.mm(e11, self.a))) / e1_all_att
            e1_e2_att = torch.exp(self.leakyrelu(torch.mm(e12, self.a))) / e1_all_att
            e1_e3_att = torch.exp(self.leakyrelu(torch.mm(e13, self.a))) / e1_all_att
            e1_e4_att = torch.exp(self.leakyrelu(torch.mm(e14, self.a))) / e1_all_att
            e2_e1_att = torch.exp(self.leakyrelu(torch.mm(e21, self.a))) / e2_all_att
            e2_e2_att = torch.exp(self.leakyrelu(torch.mm(e22, self.a))) / e2_all_att
            e2_e3_att = torch.exp(self.leakyrelu(torch.mm(e23, self.a))) / e2_all_att
            e2_e4_att = torch.exp(self.leakyrelu(torch.mm(e24, self.a))) / e2_all_att
            e3_e1_att = torch.exp(self.leakyrelu(torch.mm(e31, self.a))) / e3_all_att
            e3_e2_att = torch.exp(self.leakyrelu(torch.mm(e32, self.a))) / e3_all_att
            e3_e3_att = torch.exp(self.leakyrelu(torch.mm(e33, self.a))) / e3_all_att
            e3_e4_att = torch.exp(self.leakyrelu(torch.mm(e34, self.a))) / e3_all_att
            e4_e1_att = torch.exp(self.leakyrelu(torch.mm(e41, self.a))) / e4_all_att
            e4_e2_att = torch.exp(self.leakyrelu(torch.mm(e42, self.a))) / e4_all_att
            e4_e3_att = torch.exp(self.leakyrelu(torch.mm(e43, self.a))) / e4_all_att
            e4_e4_att = torch.exp(self.leakyrelu(torch.mm(e44, self.a))) / e4_all_att


            new_e1 = torch.mm(e1, self.W2) + torch.tanh(e11 * e1_e1_att + e12 * e1_e2_att + e13 * e1_e3_att + e14 * e1_e4_att)
            new_e2 = torch.mm(e2, self.W2) + torch.tanh(e21 * e2_e1_att + e22 * e2_e2_att + e23 * e2_e3_att + e24 * e2_e4_att)
            new_e3 = torch.mm(e3, self.W2) + torch.tanh(e31 * e3_e1_att + e32 * e3_e2_att + e33 * e3_e3_att + e34 * e3_e4_att)
            new_e4 = torch.mm(e4, self.W2) + torch.tanh(e41 * e4_e1_att + e42 * e4_e2_att + e43 * e4_e3_att + e44 * e4_e4_att)

            re1 = self.er_pos_emb(r, e1_pos)
            re2 = self.er_pos_emb(r, e2_pos)
            re3 = self.er_pos_emb(r, e3_pos)
            re4 = self.er_pos_emb(r, e4_pos)
            re1_att = torch.exp(torch.cosine_similarity(r, e1, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)))
            re2_att = torch.exp(torch.cosine_similarity(r, e2, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)))
            re3_att = torch.exp(torch.cosine_similarity(r, e3, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)))
            re4_att = torch.exp(torch.cosine_similarity(r, e4, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)))

            # re1_att = torch.exp(self.leakyrelu(re1)) / (torch.exp(self.leakyrelu(re1)) + torch.exp(self.leakyrelu(re2)) + torch.exp(self.leakyrelu(re3)) + torch.exp(self.leakyrelu(re4)))
            # re2_att = torch.exp(self.leakyrelu(re2)) / (torch.exp(self.leakyrelu(re1)) + torch.exp(self.leakyrelu(re2)) + torch.exp(self.leakyrelu(re3)) + torch.exp(self.leakyrelu(re4)))
            # re3_att = torch.exp(self.leakyrelu(re3)) / (torch.exp(self.leakyrelu(re1)) + torch.exp(self.leakyrelu(re2)) + torch.exp(self.leakyrelu(re3)) + torch.exp(self.leakyrelu(re4)))
            # re4_att = torch.exp(self.leakyrelu(re4)) / (torch.exp(self.leakyrelu(re1)) + torch.exp(self.leakyrelu(re2)) + torch.exp(self.leakyrelu(re3)) + torch.exp(self.leakyrelu(re4)))
            r = torch.mm(r, self.W3) + torch.tanh(re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1))
            # r = re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1)

            e = new_e1 * new_e2 * new_e3 * new_e4 * r


        elif batch.shape[1] == 6:
            e1 = self.E(batch[:, 1])
            e2 = self.E(batch[:, 2])
            e3 = self.E(batch[:, 3])
            e4 = self.E(batch[:, 4])
            e5 = self.E(batch[:, 5])
            e1_pos = self.convolve(r, e1, 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
            e2_pos = self.convolve(r, e2, 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
            e3_pos = self.convolve(r, e3, 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
            e4_pos = self.convolve(r, e4, 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
            e5_pos = self.convolve(r, e5, 4) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)


            e11 = torch.mm(torch.cat((e1_pos, e1_pos, r), dim=1), self.W0)
            e12 = torch.mm(torch.cat((e1_pos, e2_pos, r), dim=1), self.W0)
            e13 = torch.mm(torch.cat((e1_pos, e3_pos, r), dim=1), self.W0)
            e14 = torch.mm(torch.cat((e1_pos, e4_pos, r), dim=1), self.W0)
            e15 = torch.mm(torch.cat((e1_pos, e5_pos, r), dim=1), self.W0)
            e21 = torch.mm(torch.cat((e2_pos, e1_pos, r), dim=1), self.W0)
            e22 = torch.mm(torch.cat((e2_pos, e2_pos, r), dim=1), self.W0)
            e23 = torch.mm(torch.cat((e2_pos, e3_pos, r), dim=1), self.W0)
            e24 = torch.mm(torch.cat((e2_pos, e4_pos, r), dim=1), self.W0)
            e25 = torch.mm(torch.cat((e2_pos, e5_pos, r), dim=1), self.W0)
            e31 = torch.mm(torch.cat((e3_pos, e1_pos, r), dim=1), self.W0)
            e32 = torch.mm(torch.cat((e3_pos, e2_pos, r), dim=1), self.W0)
            e33 = torch.mm(torch.cat((e3_pos, e3_pos, r), dim=1), self.W0)
            e34 = torch.mm(torch.cat((e3_pos, e4_pos, r), dim=1), self.W0)
            e35 = torch.mm(torch.cat((e3_pos, e5_pos, r), dim=1), self.W0)
            e41 = torch.mm(torch.cat((e4_pos, e1_pos, r), dim=1), self.W0)
            e42 = torch.mm(torch.cat((e4_pos, e2_pos, r), dim=1), self.W0)
            e43 = torch.mm(torch.cat((e4_pos, e3_pos, r), dim=1), self.W0)
            e44 = torch.mm(torch.cat((e4_pos, e4_pos, r), dim=1), self.W0)
            e45 = torch.mm(torch.cat((e4_pos, e5_pos, r), dim=1), self.W0)
            e51 = torch.mm(torch.cat((e5_pos, e1_pos, r), dim=1), self.W0)
            e52 = torch.mm(torch.cat((e5_pos, e2_pos, r), dim=1), self.W0)
            e53 = torch.mm(torch.cat((e5_pos, e3_pos, r), dim=1), self.W0)
            e54 = torch.mm(torch.cat((e5_pos, e4_pos, r), dim=1), self.W0)
            e55 = torch.mm(torch.cat((e5_pos, e5_pos, r), dim=1), self.W0)

            i1 = 0
            i2 = 0
            i3 = 0
            i4 = 0
            i5 = 0
            e1_all_att = torch.zeros(e1.shape[0], 1)
            e2_all_att = torch.zeros(e2.shape[0], 1)
            e3_all_att = torch.zeros(e3.shape[0], 1)
            e4_all_att = torch.zeros(e4.shape[0], 1)
            e5_all_att = torch.zeros(e5.shape[0], 1)

            for e1_ in batch[:, 1]:
                e1_int = e1_.item()
                att_sum_1 = 0
                for hyperedge in self.dataset.inc[e1_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1:]:
                        ei = self.E(e1_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_1 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e1_all_att[i] = att_sum_1
                i += 1

            for e2_ in batch[:, 2]:
                e2_int = e2_.item()
                att_sum_2 = 0
                for hyperedge in self.dataset.inc[e2_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1:]:
                        ei = self.E(e2_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_2 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e2_all_att[i2] = att_sum_2
                i2 += 1

            for e3_ in batch[:, 3]:
                e3_int = e3_.item()
                att_sum_3 = 0
                for hyperedge in self.dataset.inc[e3_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1:]:
                        ei = self.E(e3_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_3 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e3_all_att[i3] = att_sum_3
                i3 += 1

            for e4_ in batch[:, 4]:
                e4_int = e4_.item()
                att_sum_4 = 0
                for hyperedge in self.dataset.inc[e4_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1:]:
                        ei = self.E(e4_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_4 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e4_all_att[i4] = att_sum_4
                i4 += 1

            for e5_ in batch[:, 5]:
                e5_int = e5_.item()
                att_sum_5 = 0
                for hyperedge in self.dataset.inc[e5_int]:
                    rel = hyperedge[0]
                    for ent in hyperedge[1:]:
                        ei = self.E(e5_).to(self.device)
                        ej = self.E(torch.tensor(ent).to(self.device))
                        rk = self.R(torch.tensor(rel).to(self.device))

                        cat = torch.cat((ei, ej, rk)).view(1, -1)
                        eij = torch.mm(cat, self.W0)
                        att_sum_5 += torch.exp(self.leakyrelu(torch.mm(eij, self.a)))
                e5_all_att[i5] = att_sum_5
                i5 += 1

            e1_e1_att = torch.exp(self.leakyrelu(torch.mm(e11, self.a))) / e1_all_att
            e1_e2_att = torch.exp(self.leakyrelu(torch.mm(e12, self.a))) / e1_all_att
            e1_e3_att = torch.exp(self.leakyrelu(torch.mm(e13, self.a))) / e1_all_att
            e1_e4_att = torch.exp(self.leakyrelu(torch.mm(e14, self.a))) / e1_all_att
            e1_e5_att = torch.exp(self.leakyrelu(torch.mm(e15, self.a))) / e1_all_att
            e2_e1_att = torch.exp(self.leakyrelu(torch.mm(e21, self.a))) / e2_all_att
            e2_e2_att = torch.exp(self.leakyrelu(torch.mm(e22, self.a))) / e2_all_att
            e2_e3_att = torch.exp(self.leakyrelu(torch.mm(e23, self.a))) / e2_all_att
            e2_e4_att = torch.exp(self.leakyrelu(torch.mm(e24, self.a))) / e2_all_att
            e2_e5_att = torch.exp(self.leakyrelu(torch.mm(e25, self.a))) / e2_all_att
            e3_e1_att = torch.exp(self.leakyrelu(torch.mm(e31, self.a))) / e3_all_att
            e3_e2_att = torch.exp(self.leakyrelu(torch.mm(e32, self.a))) / e3_all_att
            e3_e3_att = torch.exp(self.leakyrelu(torch.mm(e33, self.a))) / e3_all_att
            e3_e4_att = torch.exp(self.leakyrelu(torch.mm(e34, self.a))) / e3_all_att
            e3_e5_att = torch.exp(self.leakyrelu(torch.mm(e35, self.a))) / e3_all_att
            e4_e1_att = torch.exp(self.leakyrelu(torch.mm(e41, self.a))) / e4_all_att
            e4_e2_att = torch.exp(self.leakyrelu(torch.mm(e42, self.a))) / e4_all_att
            e4_e3_att = torch.exp(self.leakyrelu(torch.mm(e43, self.a))) / e4_all_att
            e4_e4_att = torch.exp(self.leakyrelu(torch.mm(e44, self.a))) / e4_all_att
            e4_e5_att = torch.exp(self.leakyrelu(torch.mm(e45, self.a))) / e4_all_att
            e5_e1_att = torch.exp(self.leakyrelu(torch.mm(e51, self.a))) / e5_all_att
            e5_e2_att = torch.exp(self.leakyrelu(torch.mm(e52, self.a))) / e5_all_att
            e5_e3_att = torch.exp(self.leakyrelu(torch.mm(e53, self.a))) / e5_all_att
            e5_e4_att = torch.exp(self.leakyrelu(torch.mm(e54, self.a))) / e5_all_att
            e5_e5_att = torch.exp(self.leakyrelu(torch.mm(e55, self.a))) / e5_all_att

            re1 = self.er_pos_emb(r, e1_pos)
            re2 = self.er_pos_emb(r, e2_pos)
            re3 = self.er_pos_emb(r, e3_pos)
            re4 = self.er_pos_emb(r, e4_pos)
            re5 = self.er_pos_emb(r, e5_pos)
            # re1_att = torch.exp(self.leakyrelu(re1)) / (torch.exp(self.leakyrelu(re1)) + torch.exp(self.leakyrelu(re2)) + torch.exp(self.leakyrelu(re3)) + torch.exp(self.leakyrelu(re4)) + torch.exp(self.leakyrelu(re5)))
            # re2_att = torch.exp(self.leakyrelu(re2)) / (torch.exp(self.leakyrelu(re1)) + torch.exp(self.leakyrelu(re2)) + torch.exp(self.leakyrelu(re3)) + torch.exp(self.leakyrelu(re4)) + torch.exp(self.leakyrelu(re5)))
            # re3_att = torch.exp(self.leakyrelu(re3)) / (torch.exp(self.leakyrelu(re1)) + torch.exp(self.leakyrelu(re2)) + torch.exp(self.leakyrelu(re3)) + torch.exp(self.leakyrelu(re4)) + torch.exp(self.leakyrelu(re5)))
            # re4_att = torch.exp(self.leakyrelu(re4)) / (torch.exp(self.leakyrelu(re1)) + torch.exp(self.leakyrelu(re2)) + torch.exp(self.leakyrelu(re3)) + torch.exp(self.leakyrelu(re4)) + torch.exp(self.leakyrelu(re5)))
            # re5_att = torch.exp(self.leakyrelu(re5)) / (torch.exp(self.leakyrelu(re1)) + torch.exp(self.leakyrelu(re2)) + torch.exp(self.leakyrelu(re3)) + torch.exp(self.leakyrelu(re4)) + torch.exp(self.leakyrelu(re5)))
            # r = re1 * re1_att + re2 * re2_att + re3 * re3_att + re4 * re4_att + re5 * re5_att
            re1_att = torch.exp(torch.cosine_similarity(r, e1, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)))
            re2_att = torch.exp(torch.cosine_similarity(r, e2, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)))
            re3_att = torch.exp(torch.cosine_similarity(r, e3, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)))
            re4_att = torch.exp(torch.cosine_similarity(r, e4, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)))
            re5_att = torch.exp(torch.cosine_similarity(r, e5, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)))

            # r = re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att
            r = torch.mm(r, self.W3) + torch.tanh(re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att.view(-1, 1))


            new_e1 = torch.mm(e1, self.W2) + torch.tanh(e11*e1_e1_att + e12*e1_e2_att + e13*e1_e3_att + e14*e1_e4_att + e15*e1_e5_att)
            new_e2 = torch.mm(e2, self.W2) + torch.tanh(e21*e2_e1_att + e22*e2_e2_att + e23*e2_e3_att + e24*e2_e4_att + e25*e2_e5_att)
            new_e3 = torch.mm(e3, self.W2) + torch.tanh(e31*e3_e1_att + e32*e3_e2_att + e33*e3_e3_att + e34*e3_e4_att + e35*e3_e5_att)
            new_e4 = torch.mm(e4, self.W2) + torch.tanh(e41*e4_e1_att + e42*e4_e2_att + e43*e4_e3_att + e44*e4_e4_att + e45*e4_e5_att)
            new_e5 = torch.mm(e5, self.W2) + torch.tanh(e51*e5_e1_att + e52*e5_e2_att + e53*e5_e3_att + e54*e5_e4_att + e55*e5_e5_att)

            e = new_e1 * new_e2 * new_e3 * new_e4 * new_e5 * r

        elif batch.shape[1] == 7:
            e1 = self.E(batch[:, 1])
            e2 = self.E(batch[:, 2])
            e3 = self.E(batch[:, 3])
            e4 = self.E(batch[:, 4])
            e5 = self.E(batch[:, 5])
            e6 = self.E(batch[:, 6])
            e1_pos = self.convolve(r, e1, 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
            e2_pos = self.convolve(r, e2, 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
            e3_pos = self.convolve(r, e3, 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
            e4_pos = self.convolve(r, e4, 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
            e5_pos = self.convolve(r, e5, 4) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
            e6_pos = self.convolve(r, e6, 5) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)

            e11 = torch.mm(torch.cat((e1_pos, e1_pos, r), dim=1), self.W0)
            e12 = torch.mm(torch.cat((e1_pos, e2_pos, r), dim=1), self.W0)
            e13 = torch.mm(torch.cat((e1_pos, e3_pos, r), dim=1), self.W0)
            e14 = torch.mm(torch.cat((e1_pos, e4_pos, r), dim=1), self.W0)
            e15 = torch.mm(torch.cat((e1_pos, e5_pos, r), dim=1), self.W0)
            e16 = torch.mm(torch.cat((e1_pos, e6_pos, r), dim=1), self.W0)
            e21 = torch.mm(torch.cat((e2_pos, e1_pos, r), dim=1), self.W0)
            e22 = torch.mm(torch.cat((e2_pos, e2_pos, r), dim=1), self.W0)
            e23 = torch.mm(torch.cat((e2_pos, e3_pos, r), dim=1), self.W0)
            e24 = torch.mm(torch.cat((e2_pos, e4_pos, r), dim=1), self.W0)
            e25 = torch.mm(torch.cat((e2_pos, e5_pos, r), dim=1), self.W0)
            e26 = torch.mm(torch.cat((e2_pos, e6_pos, r), dim=1), self.W0)
            e31 = torch.mm(torch.cat((e3_pos, e1_pos, r), dim=1), self.W0)
            e32 = torch.mm(torch.cat((e3_pos, e2_pos, r), dim=1), self.W0)
            e33 = torch.mm(torch.cat((e3_pos, e3_pos, r), dim=1), self.W0)
            e34 = torch.mm(torch.cat((e3_pos, e4_pos, r), dim=1), self.W0)
            e35 = torch.mm(torch.cat((e3_pos, e5_pos, r), dim=1), self.W0)
            e36 = torch.mm(torch.cat((e3_pos, e6_pos, r), dim=1), self.W0)
            e41 = torch.mm(torch.cat((e4_pos, e1_pos, r), dim=1), self.W0)
            e42 = torch.mm(torch.cat((e4_pos, e2_pos, r), dim=1), self.W0)
            e43 = torch.mm(torch.cat((e4_pos, e3_pos, r), dim=1), self.W0)
            e44 = torch.mm(torch.cat((e4_pos, e4_pos, r), dim=1), self.W0)
            e45 = torch.mm(torch.cat((e4_pos, e5_pos, r), dim=1), self.W0)
            e46 = torch.mm(torch.cat((e4_pos, e6_pos, r), dim=1), self.W0)
            e51 = torch.mm(torch.cat((e5_pos, e1_pos, r), dim=1), self.W0)
            e52 = torch.mm(torch.cat((e5_pos, e2_pos, r), dim=1), self.W0)
            e53 = torch.mm(torch.cat((e5_pos, e3_pos, r), dim=1), self.W0)
            e54 = torch.mm(torch.cat((e5_pos, e4_pos, r), dim=1), self.W0)
            e55 = torch.mm(torch.cat((e5_pos, e5_pos, r), dim=1), self.W0)
            e56 = torch.mm(torch.cat((e5_pos, e6_pos, r), dim=1), self.W0)
            e61 = torch.mm(torch.cat((e6_pos, e1_pos, r), dim=1), self.W0)
            e62 = torch.mm(torch.cat((e6_pos, e2_pos, r), dim=1), self.W0)
            e63 = torch.mm(torch.cat((e6_pos, e3_pos, r), dim=1), self.W0)
            e64 = torch.mm(torch.cat((e6_pos, e4_pos, r), dim=1), self.W0)
            e65 = torch.mm(torch.cat((e6_pos, e5_pos, r), dim=1), self.W0)
            e66 = torch.mm(torch.cat((e6_pos, e6_pos, r), dim=1), self.W0)

            e1_e1_att = torch.exp(self.leakyrelu(torch.mm(e11, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))))
            e1_e2_att = torch.exp(self.leakyrelu(torch.mm(e12, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))))
            e1_e3_att = torch.exp(self.leakyrelu(torch.mm(e13, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))))
            e1_e4_att = torch.exp(self.leakyrelu(torch.mm(e14, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))))
            e1_e5_att = torch.exp(self.leakyrelu(torch.mm(e15, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))))
            e1_e6_att = torch.exp(self.leakyrelu(torch.mm(e16, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))))
            e2_e1_att = torch.exp(self.leakyrelu(torch.mm(e21, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))))
            e2_e2_att = torch.exp(self.leakyrelu(torch.mm(e22, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))))
            e2_e3_att = torch.exp(self.leakyrelu(torch.mm(e23, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))))
            e2_e4_att = torch.exp(self.leakyrelu(torch.mm(e24, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))))
            e2_e5_att = torch.exp(self.leakyrelu(torch.mm(e25, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))))
            e2_e6_att = torch.exp(self.leakyrelu(torch.mm(e26, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))))
            e3_e1_att = torch.exp(self.leakyrelu(torch.mm(e31, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))))
            e3_e2_att = torch.exp(self.leakyrelu(torch.mm(e32, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))))
            e3_e3_att = torch.exp(self.leakyrelu(torch.mm(e33, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))))
            e3_e4_att = torch.exp(self.leakyrelu(torch.mm(e34, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))))
            e3_e5_att = torch.exp(self.leakyrelu(torch.mm(e35, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))))
            e3_e6_att = torch.exp(self.leakyrelu(torch.mm(e36, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))))
            e4_e1_att = torch.exp(self.leakyrelu(torch.mm(e41, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))))
            e4_e2_att = torch.exp(self.leakyrelu(torch.mm(e42, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))))
            e4_e3_att = torch.exp(self.leakyrelu(torch.mm(e43, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))))
            e4_e4_att = torch.exp(self.leakyrelu(torch.mm(e44, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))))
            e4_e5_att = torch.exp(self.leakyrelu(torch.mm(e45, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))))
            e4_e6_att = torch.exp(self.leakyrelu(torch.mm(e46, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))))
            e5_e1_att = torch.exp(self.leakyrelu(torch.mm(e51, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))))
            e5_e2_att = torch.exp(self.leakyrelu(torch.mm(e52, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))))
            e5_e3_att = torch.exp(self.leakyrelu(torch.mm(e53, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))))
            e5_e4_att = torch.exp(self.leakyrelu(torch.mm(e54, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))))
            e5_e5_att = torch.exp(self.leakyrelu(torch.mm(e55, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))))
            e5_e6_att = torch.exp(self.leakyrelu(torch.mm(e56, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))))
            e6_e1_att = torch.exp(self.leakyrelu(torch.mm(e61, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))))
            e6_e2_att = torch.exp(self.leakyrelu(torch.mm(e62, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))))
            e6_e3_att = torch.exp(self.leakyrelu(torch.mm(e63, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))))
            e6_e4_att = torch.exp(self.leakyrelu(torch.mm(e64, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))))
            e6_e5_att = torch.exp(self.leakyrelu(torch.mm(e65, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))))
            e6_e6_att = torch.exp(self.leakyrelu(torch.mm(e66, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))))


            re1 = self.er_pos_emb(r, e1_pos)
            re2 = self.er_pos_emb(r, e2_pos)
            re3 = self.er_pos_emb(r, e3_pos)
            re4 = self.er_pos_emb(r, e4_pos)
            re5 = self.er_pos_emb(r, e5_pos)
            re6 = self.er_pos_emb(r, e6_pos)

            re1_att = torch.exp(torch.cosine_similarity(r, e1, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)))
            re2_att = torch.exp(torch.cosine_similarity(r, e2, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)))
            re3_att = torch.exp(torch.cosine_similarity(r, e3, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)))
            re4_att = torch.exp(torch.cosine_similarity(r, e4, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)))
            re5_att = torch.exp(torch.cosine_similarity(r, e5, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)))
            re6_att = torch.exp(torch.cosine_similarity(r, e6, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)))

            # r = re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att + re6 * re6_att
            r = torch.mm(r, self.W3) + torch.tanh(re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att.view(-1, 1) + re6 * re6_att.view(-1, 1))


            new_e1 = torch.mm(e1, self.W2) + torch.tanh(e11*e1_e1_att + e12*e1_e2_att + e13*e1_e3_att + e14*e1_e4_att + e15*e1_e5_att + e16*e1_e6_att)
            new_e2 = torch.mm(e2, self.W2) + torch.tanh(e21*e2_e1_att + e22*e2_e2_att + e23*e2_e3_att + e24*e2_e4_att + e25*e2_e5_att + e26*e2_e6_att)
            new_e3 = torch.mm(e3, self.W2) + torch.tanh(e31*e3_e1_att + e32*e3_e2_att + e33*e3_e3_att + e34*e3_e4_att + e35*e3_e5_att + e36*e3_e6_att)
            new_e4 = torch.mm(e4, self.W2) + torch.tanh(e41*e4_e1_att + e42*e4_e2_att + e43*e4_e3_att + e44*e4_e4_att + e45*e4_e5_att + e46*e4_e6_att)
            new_e5 = torch.mm(e5, self.W2) + torch.tanh(e51*e5_e1_att + e52*e5_e2_att + e53*e5_e3_att + e54*e5_e4_att + e55*e5_e5_att + e56*e5_e6_att)
            new_e6 = torch.mm(e6, self.W2) + torch.tanh(e61*e6_e1_att + e62*e6_e2_att + e63*e6_e3_att + e64*e6_e4_att + e65*e6_e5_att + e66*e6_e6_att)

            e = new_e1 * new_e2 * new_e3 * new_e4 * new_e5 * new_e6 * r


        elif batch.shape[1] == 8:
            e1 = self.convolve(r, self.E(batch[:, 1]), 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
            e2 = self.convolve(r, self.E(batch[:, 2]), 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
            e3 = self.convolve(r, self.E(batch[:, 3]), 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
            e4 = self.convolve(r, self.E(batch[:, 4]), 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
            e5 = self.convolve(r, self.E(batch[:, 5]), 4) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
            e6 = self.convolve(r, self.E(batch[:, 6]), 5) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)
            e7 = self.convolve(r, self.E(batch[:, 7]), 6) * ms[:,6].view(-1, 1) + bs[:,6].view(-1, 1)

            e11 = torch.mm(torch.cat((e1, e1, r), dim=1), self.W0)
            e12 = torch.mm(torch.cat((e1, e2, r), dim=1), self.W0)
            e13 = torch.mm(torch.cat((e1, e3, r), dim=1), self.W0)
            e14 = torch.mm(torch.cat((e1, e4, r), dim=1), self.W0)
            e15 = torch.mm(torch.cat((e1, e5, r), dim=1), self.W0)
            e16 = torch.mm(torch.cat((e1, e6, r), dim=1), self.W0)
            e17 = torch.mm(torch.cat((e1, e7, r), dim=1), self.W0)
            e21 = torch.mm(torch.cat((e2, e1, r), dim=1), self.W0)
            e22 = torch.mm(torch.cat((e2, e2, r), dim=1), self.W0)
            e23 = torch.mm(torch.cat((e2, e3, r), dim=1), self.W0)
            e24 = torch.mm(torch.cat((e2, e4, r), dim=1), self.W0)
            e25 = torch.mm(torch.cat((e2, e5, r), dim=1), self.W0)
            e26 = torch.mm(torch.cat((e2, e6, r), dim=1), self.W0)
            e27 = torch.mm(torch.cat((e2, e7, r), dim=1), self.W0)
            e31 = torch.mm(torch.cat((e3, e1, r), dim=1), self.W0)
            e32 = torch.mm(torch.cat((e3, e2, r), dim=1), self.W0)
            e33 = torch.mm(torch.cat((e3, e3, r), dim=1), self.W0)
            e34 = torch.mm(torch.cat((e3, e4, r), dim=1), self.W0)
            e35 = torch.mm(torch.cat((e3, e5, r), dim=1), self.W0)
            e36 = torch.mm(torch.cat((e3, e6, r), dim=1), self.W0)
            e37 = torch.mm(torch.cat((e3, e7, r), dim=1), self.W0)
            e41 = torch.mm(torch.cat((e4, e1, r), dim=1), self.W0)
            e42 = torch.mm(torch.cat((e4, e2, r), dim=1), self.W0)
            e43 = torch.mm(torch.cat((e4, e3, r), dim=1), self.W0)
            e44 = torch.mm(torch.cat((e4, e4, r), dim=1), self.W0)
            e45 = torch.mm(torch.cat((e4, e5, r), dim=1), self.W0)
            e46 = torch.mm(torch.cat((e4, e6, r), dim=1), self.W0)
            e47 = torch.mm(torch.cat((e4, e7, r), dim=1), self.W0)
            e51 = torch.mm(torch.cat((e5, e1, r), dim=1), self.W0)
            e52 = torch.mm(torch.cat((e5, e2, r), dim=1), self.W0)
            e53 = torch.mm(torch.cat((e5, e3, r), dim=1), self.W0)
            e54 = torch.mm(torch.cat((e5, e4, r), dim=1), self.W0)
            e55 = torch.mm(torch.cat((e5, e5, r), dim=1), self.W0)
            e56 = torch.mm(torch.cat((e5, e6, r), dim=1), self.W0)
            e57 = torch.mm(torch.cat((e5, e7, r), dim=1), self.W0)
            e61 = torch.mm(torch.cat((e6, e1, r), dim=1), self.W0)
            e62 = torch.mm(torch.cat((e6, e2, r), dim=1), self.W0)
            e63 = torch.mm(torch.cat((e6, e3, r), dim=1), self.W0)
            e64 = torch.mm(torch.cat((e6, e4, r), dim=1), self.W0)
            e65 = torch.mm(torch.cat((e6, e5, r), dim=1), self.W0)
            e66 = torch.mm(torch.cat((e6, e6, r), dim=1), self.W0)
            e67 = torch.mm(torch.cat((e6, e7, r), dim=1), self.W0)
            e71 = torch.mm(torch.cat((e7, e1, r), dim=1), self.W0)
            e72 = torch.mm(torch.cat((e7, e2, r), dim=1), self.W0)
            e73 = torch.mm(torch.cat((e7, e3, r), dim=1), self.W0)
            e74 = torch.mm(torch.cat((e7, e4, r), dim=1), self.W0)
            e75 = torch.mm(torch.cat((e7, e5, r), dim=1), self.W0)
            e76 = torch.mm(torch.cat((e7, e6, r), dim=1), self.W0)
            e77 = torch.mm(torch.cat((e7, e7, r), dim=1), self.W0)


            e1_e1_att = torch.exp(self.leakyrelu(torch.mm(e11, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))))
            e1_e2_att = torch.exp(self.leakyrelu(torch.mm(e12, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))))
            e1_e3_att = torch.exp(self.leakyrelu(torch.mm(e13, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))))
            e1_e4_att = torch.exp(self.leakyrelu(torch.mm(e14, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))))
            e1_e5_att = torch.exp(self.leakyrelu(torch.mm(e15, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))))
            e1_e6_att = torch.exp(self.leakyrelu(torch.mm(e16, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))))
            e1_e7_att = torch.exp(self.leakyrelu(torch.mm(e17, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))))
            e2_e1_att = torch.exp(self.leakyrelu(torch.mm(e21, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))))
            e2_e2_att = torch.exp(self.leakyrelu(torch.mm(e22, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))))
            e2_e3_att = torch.exp(self.leakyrelu(torch.mm(e23, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))))
            e2_e4_att = torch.exp(self.leakyrelu(torch.mm(e24, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))))
            e2_e5_att = torch.exp(self.leakyrelu(torch.mm(e25, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))))
            e2_e6_att = torch.exp(self.leakyrelu(torch.mm(e26, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))))
            e2_e7_att = torch.exp(self.leakyrelu(torch.mm(e27, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))))
            e3_e1_att = torch.exp(self.leakyrelu(torch.mm(e31, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))))
            e3_e2_att = torch.exp(self.leakyrelu(torch.mm(e32, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))))
            e3_e3_att = torch.exp(self.leakyrelu(torch.mm(e33, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))))
            e3_e4_att = torch.exp(self.leakyrelu(torch.mm(e34, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))))
            e3_e5_att = torch.exp(self.leakyrelu(torch.mm(e35, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))))
            e3_e6_att = torch.exp(self.leakyrelu(torch.mm(e36, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))))
            e3_e7_att = torch.exp(self.leakyrelu(torch.mm(e37, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))))
            e4_e1_att = torch.exp(self.leakyrelu(torch.mm(e41, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))))
            e4_e2_att = torch.exp(self.leakyrelu(torch.mm(e42, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))))
            e4_e3_att = torch.exp(self.leakyrelu(torch.mm(e43, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))))
            e4_e4_att = torch.exp(self.leakyrelu(torch.mm(e44, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))))
            e4_e5_att = torch.exp(self.leakyrelu(torch.mm(e45, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))))
            e4_e6_att = torch.exp(self.leakyrelu(torch.mm(e46, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))))
            e4_e7_att = torch.exp(self.leakyrelu(torch.mm(e47, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))))
            e5_e1_att = torch.exp(self.leakyrelu(torch.mm(e51, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))))
            e5_e2_att = torch.exp(self.leakyrelu(torch.mm(e52, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))))
            e5_e3_att = torch.exp(self.leakyrelu(torch.mm(e53, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))))
            e5_e4_att = torch.exp(self.leakyrelu(torch.mm(e54, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))))
            e5_e5_att = torch.exp(self.leakyrelu(torch.mm(e55, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))))
            e5_e6_att = torch.exp(self.leakyrelu(torch.mm(e56, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))))
            e5_e7_att = torch.exp(self.leakyrelu(torch.mm(e57, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))))
            e6_e1_att = torch.exp(self.leakyrelu(torch.mm(e61, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))))
            e6_e2_att = torch.exp(self.leakyrelu(torch.mm(e62, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))))
            e6_e3_att = torch.exp(self.leakyrelu(torch.mm(e63, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))))
            e6_e4_att = torch.exp(self.leakyrelu(torch.mm(e64, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))))
            e6_e5_att = torch.exp(self.leakyrelu(torch.mm(e65, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))))
            e6_e6_att = torch.exp(self.leakyrelu(torch.mm(e66, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))))
            e6_e7_att = torch.exp(self.leakyrelu(torch.mm(e67, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))))
            e7_e1_att = torch.exp(self.leakyrelu(torch.mm(e71, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))))
            e7_e2_att = torch.exp(self.leakyrelu(torch.mm(e72, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))))
            e7_e3_att = torch.exp(self.leakyrelu(torch.mm(e73, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))))
            e7_e4_att = torch.exp(self.leakyrelu(torch.mm(e74, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))))
            e7_e5_att = torch.exp(self.leakyrelu(torch.mm(e75, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))))
            e7_e6_att = torch.exp(self.leakyrelu(torch.mm(e76, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))))
            e7_e7_att = torch.exp(self.leakyrelu(torch.mm(e77, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))))


            re1 = self.er_pos_emb(r, e1)
            re2 = self.er_pos_emb(r, e2)
            re3 = self.er_pos_emb(r, e3)
            re4 = self.er_pos_emb(r, e4)
            re5 = self.er_pos_emb(r, e5)
            re6 = self.er_pos_emb(r, e6)
            re7 = self.er_pos_emb(r, e7)

            re1_att = torch.exp(torch.cosine_similarity(r, e1, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)))
            re2_att = torch.exp(torch.cosine_similarity(r, e2, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)))
            re3_att = torch.exp(torch.cosine_similarity(r, e3, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)))
            re4_att = torch.exp(torch.cosine_similarity(r, e4, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)))
            re5_att = torch.exp(torch.cosine_similarity(r, e5, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)))
            re6_att = torch.exp(torch.cosine_similarity(r, e6, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)))
            re7_att = torch.exp(torch.cosine_similarity(r, e7, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)))

            # r = re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att + re6 * re6_att + re7 * re7_att
            r = torch.mm(r, self.W3) + torch.tanh(re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att.view(-1, 1) + re6 * re6_att.view(-1, 1) + re7 * re7_att.view(-1, 1))


            new_e1 = torch.mm(e1, self.W2) + torch.tanh(e11*e1_e1_att + e12*e1_e2_att + e13*e1_e3_att + e14*e1_e4_att + e15*e1_e5_att + e16*e1_e6_att + e17*e1_e7_att)
            new_e2 = torch.mm(e2, self.W2) + torch.tanh(e21*e2_e1_att + e22*e2_e2_att + e23*e2_e3_att + e24*e2_e4_att + e25*e2_e5_att + e26*e2_e6_att + e27*e2_e7_att)
            new_e3 = torch.mm(e3, self.W2) + torch.tanh(e31*e3_e1_att + e32*e3_e2_att + e33*e3_e3_att + e34*e3_e4_att + e35*e3_e5_att + e36*e3_e6_att + e37*e3_e7_att)
            new_e4 = torch.mm(e4, self.W2) + torch.tanh(e41*e4_e1_att + e42*e4_e2_att + e43*e4_e3_att + e44*e4_e4_att + e45*e4_e5_att + e46*e4_e6_att + e47*e4_e7_att)
            new_e5 = torch.mm(e5, self.W2) + torch.tanh(e51*e5_e1_att + e52*e5_e2_att + e53*e5_e3_att + e54*e5_e4_att + e55*e5_e5_att + e56*e5_e6_att + e57*e5_e7_att)
            new_e6 = torch.mm(e6, self.W2) + torch.tanh(e61*e6_e1_att + e62*e6_e2_att + e63*e6_e3_att + e64*e6_e4_att + e65*e6_e5_att + e66*e6_e6_att + e67*e6_e7_att)
            new_e7 = torch.mm(e7, self.W2) + torch.tanh(e71*e7_e1_att + e72*e7_e2_att + e73*e7_e3_att + e74*e7_e4_att + e75*e7_e5_att + e76*e7_e6_att + e77*e7_e7_att)


            e = r * new_e1 * new_e2 * new_e3 * new_e4 * new_e5 * new_e6 * new_e7



        elif batch.shape[1] == 9:
            e1 = self.convolve(r, self.E(batch[:, 1]), 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
            e2 = self.convolve(r, self.E(batch[:, 2]), 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
            e3 = self.convolve(r, self.E(batch[:, 3]), 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
            e4 = self.convolve(r, self.E(batch[:, 4]), 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
            e5 = self.convolve(r, self.E(batch[:, 5]), 4) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
            e6 = self.convolve(r, self.E(batch[:, 6]), 5) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)
            e7 = self.convolve(r, self.E(batch[:, 7]), 6) * ms[:,6].view(-1, 1) + bs[:,6].view(-1, 1)
            e8 = self.convolve(r, self.E(batch[:, 8]), 7) * ms[:,7].view(-1, 1) + bs[:,7].view(-1, 1)

            e11 = torch.mm(torch.cat((e1, e1, r), dim=1), self.W0)
            e12 = torch.mm(torch.cat((e1, e2, r), dim=1), self.W0)
            e13 = torch.mm(torch.cat((e1, e3, r), dim=1), self.W0)
            e14 = torch.mm(torch.cat((e1, e4, r), dim=1), self.W0)
            e15 = torch.mm(torch.cat((e1, e5, r), dim=1), self.W0)
            e16 = torch.mm(torch.cat((e1, e6, r), dim=1), self.W0)
            e17 = torch.mm(torch.cat((e1, e7, r), dim=1), self.W0)
            e18 = torch.mm(torch.cat((e1, e8, r), dim=1), self.W0)
            e21 = torch.mm(torch.cat((e2, e1, r), dim=1), self.W0)
            e22 = torch.mm(torch.cat((e2, e2, r), dim=1), self.W0)
            e23 = torch.mm(torch.cat((e2, e3, r), dim=1), self.W0)
            e24 = torch.mm(torch.cat((e2, e4, r), dim=1), self.W0)
            e25 = torch.mm(torch.cat((e2, e5, r), dim=1), self.W0)
            e26 = torch.mm(torch.cat((e2, e6, r), dim=1), self.W0)
            e27 = torch.mm(torch.cat((e2, e7, r), dim=1), self.W0)
            e28 = torch.mm(torch.cat((e2, e8, r), dim=1), self.W0)
            e31 = torch.mm(torch.cat((e3, e1, r), dim=1), self.W0)
            e32 = torch.mm(torch.cat((e3, e2, r), dim=1), self.W0)
            e33 = torch.mm(torch.cat((e3, e3, r), dim=1), self.W0)
            e34 = torch.mm(torch.cat((e3, e4, r), dim=1), self.W0)
            e35 = torch.mm(torch.cat((e3, e5, r), dim=1), self.W0)
            e36 = torch.mm(torch.cat((e3, e6, r), dim=1), self.W0)
            e37 = torch.mm(torch.cat((e3, e7, r), dim=1), self.W0)
            e38 = torch.mm(torch.cat((e3, e8, r), dim=1), self.W0)
            e41 = torch.mm(torch.cat((e4, e1, r), dim=1), self.W0)
            e42 = torch.mm(torch.cat((e4, e2, r), dim=1), self.W0)
            e43 = torch.mm(torch.cat((e4, e3, r), dim=1), self.W0)
            e44 = torch.mm(torch.cat((e4, e4, r), dim=1), self.W0)
            e45 = torch.mm(torch.cat((e4, e5, r), dim=1), self.W0)
            e46 = torch.mm(torch.cat((e4, e6, r), dim=1), self.W0)
            e47 = torch.mm(torch.cat((e4, e7, r), dim=1), self.W0)
            e48 = torch.mm(torch.cat((e4, e8, r), dim=1), self.W0)
            e51 = torch.mm(torch.cat((e5, e1, r), dim=1), self.W0)
            e52 = torch.mm(torch.cat((e5, e2, r), dim=1), self.W0)
            e53 = torch.mm(torch.cat((e5, e3, r), dim=1), self.W0)
            e54 = torch.mm(torch.cat((e5, e4, r), dim=1), self.W0)
            e55 = torch.mm(torch.cat((e5, e5, r), dim=1), self.W0)
            e56 = torch.mm(torch.cat((e5, e6, r), dim=1), self.W0)
            e57 = torch.mm(torch.cat((e5, e7, r), dim=1), self.W0)
            e58 = torch.mm(torch.cat((e5, e8, r), dim=1), self.W0)
            e61 = torch.mm(torch.cat((e6, e1, r), dim=1), self.W0)
            e62 = torch.mm(torch.cat((e6, e2, r), dim=1), self.W0)
            e63 = torch.mm(torch.cat((e6, e3, r), dim=1), self.W0)
            e64 = torch.mm(torch.cat((e6, e4, r), dim=1), self.W0)
            e65 = torch.mm(torch.cat((e6, e5, r), dim=1), self.W0)
            e66 = torch.mm(torch.cat((e6, e6, r), dim=1), self.W0)
            e67 = torch.mm(torch.cat((e6, e7, r), dim=1), self.W0)
            e68 = torch.mm(torch.cat((e6, e8, r), dim=1), self.W0)
            e71 = torch.mm(torch.cat((e7, e1, r), dim=1), self.W0)
            e72 = torch.mm(torch.cat((e7, e2, r), dim=1), self.W0)
            e73 = torch.mm(torch.cat((e7, e3, r), dim=1), self.W0)
            e74 = torch.mm(torch.cat((e7, e4, r), dim=1), self.W0)
            e75 = torch.mm(torch.cat((e7, e5, r), dim=1), self.W0)
            e76 = torch.mm(torch.cat((e7, e6, r), dim=1), self.W0)
            e77 = torch.mm(torch.cat((e7, e7, r), dim=1), self.W0)
            e78 = torch.mm(torch.cat((e7, e8, r), dim=1), self.W0)
            e81 = torch.mm(torch.cat((e8, e1, r), dim=1), self.W0)
            e82 = torch.mm(torch.cat((e8, e2, r), dim=1), self.W0)
            e83 = torch.mm(torch.cat((e8, e3, r), dim=1), self.W0)
            e84 = torch.mm(torch.cat((e8, e4, r), dim=1), self.W0)
            e85 = torch.mm(torch.cat((e8, e5, r), dim=1), self.W0)
            e86 = torch.mm(torch.cat((e8, e6, r), dim=1), self.W0)
            e87 = torch.mm(torch.cat((e8, e7, r), dim=1), self.W0)
            e88 = torch.mm(torch.cat((e8, e8, r), dim=1), self.W0)


            e1_e1_att = torch.exp(self.leakyrelu(torch.mm(e11, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))))
            e1_e2_att = torch.exp(self.leakyrelu(torch.mm(e12, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))))
            e1_e3_att = torch.exp(self.leakyrelu(torch.mm(e13, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))))
            e1_e4_att = torch.exp(self.leakyrelu(torch.mm(e14, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))))
            e1_e5_att = torch.exp(self.leakyrelu(torch.mm(e15, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))))
            e1_e6_att = torch.exp(self.leakyrelu(torch.mm(e16, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))))
            e1_e7_att = torch.exp(self.leakyrelu(torch.mm(e17, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))))
            e1_e8_att = torch.exp(self.leakyrelu(torch.mm(e18, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))))
            e2_e1_att = torch.exp(self.leakyrelu(torch.mm(e21, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))))
            e2_e2_att = torch.exp(self.leakyrelu(torch.mm(e22, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))))
            e2_e3_att = torch.exp(self.leakyrelu(torch.mm(e23, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))))
            e2_e4_att = torch.exp(self.leakyrelu(torch.mm(e24, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))))
            e2_e5_att = torch.exp(self.leakyrelu(torch.mm(e25, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))))
            e2_e6_att = torch.exp(self.leakyrelu(torch.mm(e26, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))))
            e2_e7_att = torch.exp(self.leakyrelu(torch.mm(e27, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))))
            e2_e8_att = torch.exp(self.leakyrelu(torch.mm(e28, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))))
            e3_e1_att = torch.exp(self.leakyrelu(torch.mm(e31, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))))
            e3_e2_att = torch.exp(self.leakyrelu(torch.mm(e32, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))))
            e3_e3_att = torch.exp(self.leakyrelu(torch.mm(e33, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))))
            e3_e4_att = torch.exp(self.leakyrelu(torch.mm(e34, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))))
            e3_e5_att = torch.exp(self.leakyrelu(torch.mm(e35, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))))
            e3_e6_att = torch.exp(self.leakyrelu(torch.mm(e36, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))))
            e3_e7_att = torch.exp(self.leakyrelu(torch.mm(e37, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))))
            e3_e8_att = torch.exp(self.leakyrelu(torch.mm(e38, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))))
            e4_e1_att = torch.exp(self.leakyrelu(torch.mm(e41, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))))
            e4_e2_att = torch.exp(self.leakyrelu(torch.mm(e42, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))))
            e4_e3_att = torch.exp(self.leakyrelu(torch.mm(e43, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))))
            e4_e4_att = torch.exp(self.leakyrelu(torch.mm(e44, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))))
            e4_e5_att = torch.exp(self.leakyrelu(torch.mm(e45, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))))
            e4_e6_att = torch.exp(self.leakyrelu(torch.mm(e46, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))))
            e4_e7_att = torch.exp(self.leakyrelu(torch.mm(e47, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))))
            e4_e8_att = torch.exp(self.leakyrelu(torch.mm(e48, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))))
            e5_e1_att = torch.exp(self.leakyrelu(torch.mm(e51, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))))
            e5_e2_att = torch.exp(self.leakyrelu(torch.mm(e52, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))))
            e5_e3_att = torch.exp(self.leakyrelu(torch.mm(e53, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))))
            e5_e4_att = torch.exp(self.leakyrelu(torch.mm(e54, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))))
            e5_e5_att = torch.exp(self.leakyrelu(torch.mm(e55, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))))
            e5_e6_att = torch.exp(self.leakyrelu(torch.mm(e56, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))))
            e5_e7_att = torch.exp(self.leakyrelu(torch.mm(e57, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))))
            e5_e8_att = torch.exp(self.leakyrelu(torch.mm(e58, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))))
            e6_e1_att = torch.exp(self.leakyrelu(torch.mm(e61, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))))
            e6_e2_att = torch.exp(self.leakyrelu(torch.mm(e62, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))))
            e6_e3_att = torch.exp(self.leakyrelu(torch.mm(e63, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))))
            e6_e4_att = torch.exp(self.leakyrelu(torch.mm(e64, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))))
            e6_e5_att = torch.exp(self.leakyrelu(torch.mm(e65, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))))
            e6_e6_att = torch.exp(self.leakyrelu(torch.mm(e66, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))))
            e6_e7_att = torch.exp(self.leakyrelu(torch.mm(e67, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))))
            e6_e8_att = torch.exp(self.leakyrelu(torch.mm(e68, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))))
            e7_e1_att = torch.exp(self.leakyrelu(torch.mm(e71, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))))
            e7_e2_att = torch.exp(self.leakyrelu(torch.mm(e72, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))))
            e7_e3_att = torch.exp(self.leakyrelu(torch.mm(e73, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))))
            e7_e4_att = torch.exp(self.leakyrelu(torch.mm(e74, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))))
            e7_e5_att = torch.exp(self.leakyrelu(torch.mm(e75, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))))
            e7_e6_att = torch.exp(self.leakyrelu(torch.mm(e76, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))))
            e7_e7_att = torch.exp(self.leakyrelu(torch.mm(e77, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))))
            e7_e8_att = torch.exp(self.leakyrelu(torch.mm(e78, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))))
            e8_e1_att = torch.exp(self.leakyrelu(torch.mm(e81, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))))
            e8_e2_att = torch.exp(self.leakyrelu(torch.mm(e82, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))))
            e8_e3_att = torch.exp(self.leakyrelu(torch.mm(e83, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))))
            e8_e4_att = torch.exp(self.leakyrelu(torch.mm(e84, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))))
            e8_e5_att = torch.exp(self.leakyrelu(torch.mm(e85, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))))
            e8_e6_att = torch.exp(self.leakyrelu(torch.mm(e86, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))))
            e8_e7_att = torch.exp(self.leakyrelu(torch.mm(e87, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))))
            e8_e8_att = torch.exp(self.leakyrelu(torch.mm(e88, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))))


            re1 = self.er_pos_emb(r, e1)
            re2 = self.er_pos_emb(r, e2)
            re3 = self.er_pos_emb(r, e3)
            re4 = self.er_pos_emb(r, e4)
            re5 = self.er_pos_emb(r, e5)
            re6 = self.er_pos_emb(r, e6)
            re7 = self.er_pos_emb(r, e7)
            re8 = self.er_pos_emb(r, e8)
            re1_att = torch.exp(torch.cosine_similarity(r, e1, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)))
            re2_att = torch.exp(torch.cosine_similarity(r, e2, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)))
            re3_att = torch.exp(torch.cosine_similarity(r, e3, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)))
            re4_att = torch.exp(torch.cosine_similarity(r, e4, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)))
            re5_att = torch.exp(torch.cosine_similarity(r, e5, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)))
            re6_att = torch.exp(torch.cosine_similarity(r, e6, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)))
            re7_att = torch.exp(torch.cosine_similarity(r, e7, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)))
            re8_att = torch.exp(torch.cosine_similarity(r, e8, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)))

            # r = re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att + re6 * re6_att + re7 * re7_att + re8 * re8_att
            r = torch.mm(r, self.W3) + torch.tanh(re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att.view(-1, 1) + re6 * re6_att.view(-1, 1) + re7 * re7_att.view(-1, 1) + re8 * re8_att.view(-1, 1))

            new_e1 = torch.mm(e1, self.W2) + torch.tanh(e11*e1_e1_att + e12*e1_e2_att + e13*e1_e3_att + e14*e1_e4_att + e15*e1_e5_att + e16*e1_e6_att + e17*e1_e7_att + e18*e1_e8_att)
            new_e2 = torch.mm(e2, self.W2) + torch.tanh(e21*e2_e1_att + e22*e2_e2_att + e23*e2_e3_att + e24*e2_e4_att + e25*e2_e5_att + e26*e2_e6_att + e27*e2_e7_att + e28*e2_e8_att)
            new_e3 = torch.mm(e3, self.W2) + torch.tanh(e31*e3_e1_att + e32*e3_e2_att + e33*e3_e3_att + e34*e3_e4_att + e35*e3_e5_att + e36*e3_e6_att + e37*e3_e7_att + e38*e3_e8_att)
            new_e4 = torch.mm(e4, self.W2) + torch.tanh(e41*e4_e1_att + e42*e4_e2_att + e43*e4_e3_att + e44*e4_e4_att + e45*e4_e5_att + e46*e4_e6_att + e47*e4_e7_att + e48*e4_e8_att)
            new_e5 = torch.mm(e5, self.W2) + torch.tanh(e51*e5_e1_att + e52*e5_e2_att + e53*e5_e3_att + e54*e5_e4_att + e55*e5_e5_att + e56*e5_e6_att + e57*e5_e7_att + e58*e5_e8_att)
            new_e6 = torch.mm(e6, self.W2) + torch.tanh(e61*e6_e1_att + e62*e6_e2_att + e63*e6_e3_att + e64*e6_e4_att + e65*e6_e5_att + e66*e6_e6_att + e67*e6_e7_att + e68*e6_e8_att)
            new_e7 = torch.mm(e7, self.W2) + torch.tanh(e71*e7_e1_att + e72*e7_e2_att + e73*e7_e3_att + e74*e7_e4_att + e75*e7_e5_att + e76*e7_e6_att + e77*e7_e7_att + e78*e7_e8_att)
            new_e8 = torch.mm(e8, self.W2) + torch.tanh(e81*e8_e1_att + e82*e8_e2_att + e83*e8_e3_att + e84*e8_e4_att + e85*e8_e5_att + e86*e8_e6_att + e87*e8_e7_att + e88*e8_e8_att)

            e = r * new_e1 * new_e2 * new_e3 * new_e4 * new_e5 * new_e6 * new_e7 * new_e8


        elif batch.shape[1] == 10:
            e1 = self.convolve(r, self.E(batch[:, 1]), 0) * ms[:,0].view(-1, 1) + bs[:,0].view(-1, 1)
            e2 = self.convolve(r, self.E(batch[:, 2]), 1) * ms[:,1].view(-1, 1) + bs[:,1].view(-1, 1)
            e3 = self.convolve(r, self.E(batch[:, 3]), 2) * ms[:,2].view(-1, 1) + bs[:,2].view(-1, 1)
            e4 = self.convolve(r, self.E(batch[:, 4]), 3) * ms[:,3].view(-1, 1) + bs[:,3].view(-1, 1)
            e5 = self.convolve(r, self.E(batch[:, 5]), 4) * ms[:,4].view(-1, 1) + bs[:,4].view(-1, 1)
            e6 = self.convolve(r, self.E(batch[:, 6]), 5) * ms[:,5].view(-1, 1) + bs[:,5].view(-1, 1)
            e7 = self.convolve(r, self.E(batch[:, 7]), 6) * ms[:,6].view(-1, 1) + bs[:,6].view(-1, 1)
            e8 = self.convolve(r, self.E(batch[:, 8]), 7) * ms[:,7].view(-1, 1) + bs[:,7].view(-1, 1)
            e9 = self.convolve(r, self.E(batch[:, 9]), 8) * ms[:,8].view(-1, 1) + bs[:,8].view(-1, 1)


            e11 = torch.mm(torch.cat((e1, e1, r), dim=1), self.W0)
            e12 = torch.mm(torch.cat((e1, e2, r), dim=1), self.W0)
            e13 = torch.mm(torch.cat((e1, e3, r), dim=1), self.W0)
            e14 = torch.mm(torch.cat((e1, e4, r), dim=1), self.W0)
            e15 = torch.mm(torch.cat((e1, e5, r), dim=1), self.W0)
            e16 = torch.mm(torch.cat((e1, e6, r), dim=1), self.W0)
            e17 = torch.mm(torch.cat((e1, e7, r), dim=1), self.W0)
            e18 = torch.mm(torch.cat((e1, e8, r), dim=1), self.W0)
            e19 = torch.mm(torch.cat((e1, e9, r), dim=1), self.W0)
            e21 = torch.mm(torch.cat((e2, e1, r), dim=1), self.W0)
            e22 = torch.mm(torch.cat((e2, e2, r), dim=1), self.W0)
            e23 = torch.mm(torch.cat((e2, e3, r), dim=1), self.W0)
            e24 = torch.mm(torch.cat((e2, e4, r), dim=1), self.W0)
            e25 = torch.mm(torch.cat((e2, e5, r), dim=1), self.W0)
            e26 = torch.mm(torch.cat((e2, e6, r), dim=1), self.W0)
            e27 = torch.mm(torch.cat((e2, e7, r), dim=1), self.W0)
            e28 = torch.mm(torch.cat((e2, e8, r), dim=1), self.W0)
            e29 = torch.mm(torch.cat((e2, e9, r), dim=1), self.W0)
            e31 = torch.mm(torch.cat((e3, e1, r), dim=1), self.W0)
            e32 = torch.mm(torch.cat((e3, e2, r), dim=1), self.W0)
            e33 = torch.mm(torch.cat((e3, e3, r), dim=1), self.W0)
            e34 = torch.mm(torch.cat((e3, e4, r), dim=1), self.W0)
            e35 = torch.mm(torch.cat((e3, e5, r), dim=1), self.W0)
            e36 = torch.mm(torch.cat((e3, e6, r), dim=1), self.W0)
            e37 = torch.mm(torch.cat((e3, e7, r), dim=1), self.W0)
            e38 = torch.mm(torch.cat((e3, e8, r), dim=1), self.W0)
            e39 = torch.mm(torch.cat((e3, e9, r), dim=1), self.W0)
            e41 = torch.mm(torch.cat((e4, e1, r), dim=1), self.W0)
            e42 = torch.mm(torch.cat((e4, e2, r), dim=1), self.W0)
            e43 = torch.mm(torch.cat((e4, e3, r), dim=1), self.W0)
            e44 = torch.mm(torch.cat((e4, e4, r), dim=1), self.W0)
            e45 = torch.mm(torch.cat((e4, e5, r), dim=1), self.W0)
            e46 = torch.mm(torch.cat((e4, e6, r), dim=1), self.W0)
            e47 = torch.mm(torch.cat((e4, e7, r), dim=1), self.W0)
            e48 = torch.mm(torch.cat((e4, e8, r), dim=1), self.W0)
            e49 = torch.mm(torch.cat((e4, e9, r), dim=1), self.W0)
            e51 = torch.mm(torch.cat((e5, e1, r), dim=1), self.W0)
            e52 = torch.mm(torch.cat((e5, e2, r), dim=1), self.W0)
            e53 = torch.mm(torch.cat((e5, e3, r), dim=1), self.W0)
            e54 = torch.mm(torch.cat((e5, e4, r), dim=1), self.W0)
            e55 = torch.mm(torch.cat((e5, e5, r), dim=1), self.W0)
            e56 = torch.mm(torch.cat((e5, e6, r), dim=1), self.W0)
            e57 = torch.mm(torch.cat((e5, e7, r), dim=1), self.W0)
            e58 = torch.mm(torch.cat((e5, e8, r), dim=1), self.W0)
            e59 = torch.mm(torch.cat((e5, e9, r), dim=1), self.W0)
            e61 = torch.mm(torch.cat((e6, e1, r), dim=1), self.W0)
            e62 = torch.mm(torch.cat((e6, e2, r), dim=1), self.W0)
            e63 = torch.mm(torch.cat((e6, e3, r), dim=1), self.W0)
            e64 = torch.mm(torch.cat((e6, e4, r), dim=1), self.W0)
            e65 = torch.mm(torch.cat((e6, e5, r), dim=1), self.W0)
            e66 = torch.mm(torch.cat((e6, e6, r), dim=1), self.W0)
            e67 = torch.mm(torch.cat((e6, e7, r), dim=1), self.W0)
            e68 = torch.mm(torch.cat((e6, e8, r), dim=1), self.W0)
            e69 = torch.mm(torch.cat((e6, e9, r), dim=1), self.W0)
            e71 = torch.mm(torch.cat((e7, e1, r), dim=1), self.W0)
            e72 = torch.mm(torch.cat((e7, e2, r), dim=1), self.W0)
            e73 = torch.mm(torch.cat((e7, e3, r), dim=1), self.W0)
            e74 = torch.mm(torch.cat((e7, e4, r), dim=1), self.W0)
            e75 = torch.mm(torch.cat((e7, e5, r), dim=1), self.W0)
            e76 = torch.mm(torch.cat((e7, e6, r), dim=1), self.W0)
            e77 = torch.mm(torch.cat((e7, e7, r), dim=1), self.W0)
            e78 = torch.mm(torch.cat((e7, e8, r), dim=1), self.W0)
            e79 = torch.mm(torch.cat((e7, e9, r), dim=1), self.W0)
            e81 = torch.mm(torch.cat((e8, e1, r), dim=1), self.W0)
            e82 = torch.mm(torch.cat((e8, e2, r), dim=1), self.W0)
            e83 = torch.mm(torch.cat((e8, e3, r), dim=1), self.W0)
            e84 = torch.mm(torch.cat((e8, e4, r), dim=1), self.W0)
            e85 = torch.mm(torch.cat((e8, e5, r), dim=1), self.W0)
            e86 = torch.mm(torch.cat((e8, e6, r), dim=1), self.W0)
            e87 = torch.mm(torch.cat((e8, e7, r), dim=1), self.W0)
            e88 = torch.mm(torch.cat((e8, e8, r), dim=1), self.W0)
            e89 = torch.mm(torch.cat((e8, e9, r), dim=1), self.W0)
            e91 = torch.mm(torch.cat((e9, e1, r), dim=1), self.W0)
            e92 = torch.mm(torch.cat((e9, e2, r), dim=1), self.W0)
            e93 = torch.mm(torch.cat((e9, e3, r), dim=1), self.W0)
            e94 = torch.mm(torch.cat((e9, e4, r), dim=1), self.W0)
            e95 = torch.mm(torch.cat((e9, e5, r), dim=1), self.W0)
            e96 = torch.mm(torch.cat((e9, e6, r), dim=1), self.W0)
            e97 = torch.mm(torch.cat((e9, e7, r), dim=1), self.W0)
            e98 = torch.mm(torch.cat((e9, e8, r), dim=1), self.W0)
            e99 = torch.mm(torch.cat((e9, e9, r), dim=1), self.W0)


            e1_e2_att = torch.exp(self.leakyrelu(torch.mm(e12, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))) + torch.exp(self.leakyrelu(torch.mm(e19, self.a))))
            e1_e1_att = torch.exp(self.leakyrelu(torch.mm(e11, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))) + torch.exp(self.leakyrelu(torch.mm(e19, self.a))))
            e1_e3_att = torch.exp(self.leakyrelu(torch.mm(e13, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))) + torch.exp(self.leakyrelu(torch.mm(e19, self.a))))
            e1_e4_att = torch.exp(self.leakyrelu(torch.mm(e14, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))) + torch.exp(self.leakyrelu(torch.mm(e19, self.a))))
            e1_e5_att = torch.exp(self.leakyrelu(torch.mm(e15, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))) + torch.exp(self.leakyrelu(torch.mm(e19, self.a))))
            e1_e6_att = torch.exp(self.leakyrelu(torch.mm(e16, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))) + torch.exp(self.leakyrelu(torch.mm(e19, self.a))))
            e1_e7_att = torch.exp(self.leakyrelu(torch.mm(e17, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))) + torch.exp(self.leakyrelu(torch.mm(e19, self.a))))
            e1_e8_att = torch.exp(self.leakyrelu(torch.mm(e18, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))) + torch.exp(self.leakyrelu(torch.mm(e19, self.a))))
            e1_e9_att = torch.exp(self.leakyrelu(torch.mm(e19, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e11, self.a))) + torch.exp(self.leakyrelu(torch.mm(e12, self.a))) + torch.exp(self.leakyrelu(torch.mm(e13, self.a))) + torch.exp(self.leakyrelu(torch.mm(e14, self.a))) + torch.exp(self.leakyrelu(torch.mm(e15, self.a))) + torch.exp(self.leakyrelu(torch.mm(e16, self.a))) + torch.exp(self.leakyrelu(torch.mm(e17, self.a))) + torch.exp(self.leakyrelu(torch.mm(e18, self.a))) + torch.exp(self.leakyrelu(torch.mm(e19, self.a))))
            e2_e1_att = torch.exp(self.leakyrelu(torch.mm(e21, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))) + torch.exp(self.leakyrelu(torch.mm(e29, self.a))))
            e2_e2_att = torch.exp(self.leakyrelu(torch.mm(e22, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))) + torch.exp(self.leakyrelu(torch.mm(e29, self.a))))
            e2_e3_att = torch.exp(self.leakyrelu(torch.mm(e23, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))) + torch.exp(self.leakyrelu(torch.mm(e29, self.a))))
            e2_e4_att = torch.exp(self.leakyrelu(torch.mm(e24, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))) + torch.exp(self.leakyrelu(torch.mm(e29, self.a))))
            e2_e5_att = torch.exp(self.leakyrelu(torch.mm(e25, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))) + torch.exp(self.leakyrelu(torch.mm(e29, self.a))))
            e2_e6_att = torch.exp(self.leakyrelu(torch.mm(e26, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))) + torch.exp(self.leakyrelu(torch.mm(e29, self.a))))
            e2_e7_att = torch.exp(self.leakyrelu(torch.mm(e27, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))) + torch.exp(self.leakyrelu(torch.mm(e29, self.a))))
            e2_e8_att = torch.exp(self.leakyrelu(torch.mm(e28, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))) + torch.exp(self.leakyrelu(torch.mm(e29, self.a))))
            e2_e9_att = torch.exp(self.leakyrelu(torch.mm(e29, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e21, self.a))) + torch.exp(self.leakyrelu(torch.mm(e22, self.a))) + torch.exp(self.leakyrelu(torch.mm(e23, self.a))) + torch.exp(self.leakyrelu(torch.mm(e24, self.a))) + torch.exp(self.leakyrelu(torch.mm(e25, self.a))) + torch.exp(self.leakyrelu(torch.mm(e26, self.a))) + torch.exp(self.leakyrelu(torch.mm(e27, self.a))) + torch.exp(self.leakyrelu(torch.mm(e28, self.a))) + torch.exp(self.leakyrelu(torch.mm(e29, self.a))))
            e3_e1_att = torch.exp(self.leakyrelu(torch.mm(e31, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))) + torch.exp(self.leakyrelu(torch.mm(e39, self.a))))
            e3_e2_att = torch.exp(self.leakyrelu(torch.mm(e32, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))) + torch.exp(self.leakyrelu(torch.mm(e39, self.a))))
            e3_e3_att = torch.exp(self.leakyrelu(torch.mm(e33, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))) + torch.exp(self.leakyrelu(torch.mm(e39, self.a))))
            e3_e4_att = torch.exp(self.leakyrelu(torch.mm(e34, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))) + torch.exp(self.leakyrelu(torch.mm(e39, self.a))))
            e3_e5_att = torch.exp(self.leakyrelu(torch.mm(e35, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))) + torch.exp(self.leakyrelu(torch.mm(e39, self.a))))
            e3_e6_att = torch.exp(self.leakyrelu(torch.mm(e36, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))) + torch.exp(self.leakyrelu(torch.mm(e39, self.a))))
            e3_e7_att = torch.exp(self.leakyrelu(torch.mm(e37, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))) + torch.exp(self.leakyrelu(torch.mm(e39, self.a))))
            e3_e8_att = torch.exp(self.leakyrelu(torch.mm(e38, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))) + torch.exp(self.leakyrelu(torch.mm(e39, self.a))))
            e3_e9_att = torch.exp(self.leakyrelu(torch.mm(e39, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e31, self.a))) + torch.exp(self.leakyrelu(torch.mm(e32, self.a))) + torch.exp(self.leakyrelu(torch.mm(e33, self.a))) + torch.exp(self.leakyrelu(torch.mm(e34, self.a))) + torch.exp(self.leakyrelu(torch.mm(e35, self.a))) + torch.exp(self.leakyrelu(torch.mm(e36, self.a))) + torch.exp(self.leakyrelu(torch.mm(e37, self.a))) + torch.exp(self.leakyrelu(torch.mm(e38, self.a))) + torch.exp(self.leakyrelu(torch.mm(e39, self.a))))
            e4_e1_att = torch.exp(self.leakyrelu(torch.mm(e41, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))) + torch.exp(self.leakyrelu(torch.mm(e49, self.a))))
            e4_e2_att = torch.exp(self.leakyrelu(torch.mm(e42, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))) + torch.exp(self.leakyrelu(torch.mm(e49, self.a))))
            e4_e3_att = torch.exp(self.leakyrelu(torch.mm(e43, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))) + torch.exp(self.leakyrelu(torch.mm(e49, self.a))))
            e4_e4_att = torch.exp(self.leakyrelu(torch.mm(e44, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))) + torch.exp(self.leakyrelu(torch.mm(e49, self.a))))
            e4_e5_att = torch.exp(self.leakyrelu(torch.mm(e45, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))) + torch.exp(self.leakyrelu(torch.mm(e49, self.a))))
            e4_e6_att = torch.exp(self.leakyrelu(torch.mm(e46, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))) + torch.exp(self.leakyrelu(torch.mm(e49, self.a))))
            e4_e7_att = torch.exp(self.leakyrelu(torch.mm(e47, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))) + torch.exp(self.leakyrelu(torch.mm(e49, self.a))))
            e4_e8_att = torch.exp(self.leakyrelu(torch.mm(e48, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))) + torch.exp(self.leakyrelu(torch.mm(e49, self.a))))
            e4_e9_att = torch.exp(self.leakyrelu(torch.mm(e49, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e41, self.a))) + torch.exp(self.leakyrelu(torch.mm(e42, self.a))) + torch.exp(self.leakyrelu(torch.mm(e43, self.a))) + torch.exp(self.leakyrelu(torch.mm(e44, self.a))) + torch.exp(self.leakyrelu(torch.mm(e45, self.a))) + torch.exp(self.leakyrelu(torch.mm(e46, self.a))) + torch.exp(self.leakyrelu(torch.mm(e47, self.a))) + torch.exp(self.leakyrelu(torch.mm(e48, self.a))) + torch.exp(self.leakyrelu(torch.mm(e49, self.a))))
            e5_e1_att = torch.exp(self.leakyrelu(torch.mm(e51, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))) + torch.exp(self.leakyrelu(torch.mm(e59, self.a))))
            e5_e2_att = torch.exp(self.leakyrelu(torch.mm(e52, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))) + torch.exp(self.leakyrelu(torch.mm(e59, self.a))))
            e5_e3_att = torch.exp(self.leakyrelu(torch.mm(e53, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))) + torch.exp(self.leakyrelu(torch.mm(e59, self.a))))
            e5_e4_att = torch.exp(self.leakyrelu(torch.mm(e54, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))) + torch.exp(self.leakyrelu(torch.mm(e59, self.a))))
            e5_e5_att = torch.exp(self.leakyrelu(torch.mm(e55, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))) + torch.exp(self.leakyrelu(torch.mm(e59, self.a))))
            e5_e6_att = torch.exp(self.leakyrelu(torch.mm(e56, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))) + torch.exp(self.leakyrelu(torch.mm(e59, self.a))))
            e5_e7_att = torch.exp(self.leakyrelu(torch.mm(e57, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))) + torch.exp(self.leakyrelu(torch.mm(e59, self.a))))
            e5_e8_att = torch.exp(self.leakyrelu(torch.mm(e58, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))) + torch.exp(self.leakyrelu(torch.mm(e59, self.a))))
            e5_e9_att = torch.exp(self.leakyrelu(torch.mm(e59, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e51, self.a))) + torch.exp(self.leakyrelu(torch.mm(e52, self.a))) + torch.exp(self.leakyrelu(torch.mm(e53, self.a))) + torch.exp(self.leakyrelu(torch.mm(e54, self.a))) + torch.exp(self.leakyrelu(torch.mm(e55, self.a))) + torch.exp(self.leakyrelu(torch.mm(e56, self.a))) + torch.exp(self.leakyrelu(torch.mm(e57, self.a))) + torch.exp(self.leakyrelu(torch.mm(e58, self.a))) + torch.exp(self.leakyrelu(torch.mm(e59, self.a))))
            e6_e1_att = torch.exp(self.leakyrelu(torch.mm(e61, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e6_e2_att = torch.exp(self.leakyrelu(torch.mm(e62, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e6_e3_att = torch.exp(self.leakyrelu(torch.mm(e63, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e6_e4_att = torch.exp(self.leakyrelu(torch.mm(e64, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e6_e5_att = torch.exp(self.leakyrelu(torch.mm(e65, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e6_e6_att = torch.exp(self.leakyrelu(torch.mm(e66, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e6_e7_att = torch.exp(self.leakyrelu(torch.mm(e67, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e6_e8_att = torch.exp(self.leakyrelu(torch.mm(e68, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e6_e9_att = torch.exp(self.leakyrelu(torch.mm(e69, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e61, self.a))) + torch.exp(self.leakyrelu(torch.mm(e62, self.a))) + torch.exp(self.leakyrelu(torch.mm(e63, self.a))) + torch.exp(self.leakyrelu(torch.mm(e64, self.a))) + torch.exp(self.leakyrelu(torch.mm(e65, self.a))) + torch.exp(self.leakyrelu(torch.mm(e66, self.a))) + torch.exp(self.leakyrelu(torch.mm(e67, self.a))) + torch.exp(self.leakyrelu(torch.mm(e68, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e7_e1_att = torch.exp(self.leakyrelu(torch.mm(e71, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))) + torch.exp(self.leakyrelu(torch.mm(e79, self.a))))
            e7_e2_att = torch.exp(self.leakyrelu(torch.mm(e72, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))) + torch.exp(self.leakyrelu(torch.mm(e79, self.a))))
            e7_e3_att = torch.exp(self.leakyrelu(torch.mm(e73, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))) + torch.exp(self.leakyrelu(torch.mm(e79, self.a))))
            e7_e4_att = torch.exp(self.leakyrelu(torch.mm(e74, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))) + torch.exp(self.leakyrelu(torch.mm(e79, self.a))))
            e7_e5_att = torch.exp(self.leakyrelu(torch.mm(e75, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))) + torch.exp(self.leakyrelu(torch.mm(e79, self.a))))
            e7_e6_att = torch.exp(self.leakyrelu(torch.mm(e76, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))) + torch.exp(self.leakyrelu(torch.mm(e79, self.a))))
            e7_e7_att = torch.exp(self.leakyrelu(torch.mm(e77, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))) + torch.exp(self.leakyrelu(torch.mm(e79, self.a))))
            e7_e8_att = torch.exp(self.leakyrelu(torch.mm(e78, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))) + torch.exp(self.leakyrelu(torch.mm(e79, self.a))))
            e7_e9_att = torch.exp(self.leakyrelu(torch.mm(e79, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e71, self.a))) + torch.exp(self.leakyrelu(torch.mm(e72, self.a))) + torch.exp(self.leakyrelu(torch.mm(e73, self.a))) + torch.exp(self.leakyrelu(torch.mm(e74, self.a))) + torch.exp(self.leakyrelu(torch.mm(e75, self.a))) + torch.exp(self.leakyrelu(torch.mm(e76, self.a))) + torch.exp(self.leakyrelu(torch.mm(e77, self.a))) + torch.exp(self.leakyrelu(torch.mm(e78, self.a))) + torch.exp(self.leakyrelu(torch.mm(e79, self.a))))
            e8_e1_att = torch.exp(self.leakyrelu(torch.mm(e81, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))) + torch.exp(self.leakyrelu(torch.mm(e69, self.a))))
            e8_e2_att = torch.exp(self.leakyrelu(torch.mm(e82, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))) + torch.exp(self.leakyrelu(torch.mm(e89, self.a))))
            e8_e3_att = torch.exp(self.leakyrelu(torch.mm(e83, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))) + torch.exp(self.leakyrelu(torch.mm(e89, self.a))))
            e8_e4_att = torch.exp(self.leakyrelu(torch.mm(e84, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))) + torch.exp(self.leakyrelu(torch.mm(e89, self.a))))
            e8_e5_att = torch.exp(self.leakyrelu(torch.mm(e85, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))) + torch.exp(self.leakyrelu(torch.mm(e89, self.a))))
            e8_e6_att = torch.exp(self.leakyrelu(torch.mm(e86, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))) + torch.exp(self.leakyrelu(torch.mm(e89, self.a))))
            e8_e7_att = torch.exp(self.leakyrelu(torch.mm(e87, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))) + torch.exp(self.leakyrelu(torch.mm(e89, self.a))))
            e8_e8_att = torch.exp(self.leakyrelu(torch.mm(e88, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))) + torch.exp(self.leakyrelu(torch.mm(e89, self.a))))
            e8_e9_att = torch.exp(self.leakyrelu(torch.mm(e89, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e81, self.a))) + torch.exp(self.leakyrelu(torch.mm(e82, self.a))) + torch.exp(self.leakyrelu(torch.mm(e83, self.a))) + torch.exp(self.leakyrelu(torch.mm(e84, self.a))) + torch.exp(self.leakyrelu(torch.mm(e85, self.a))) + torch.exp(self.leakyrelu(torch.mm(e86, self.a))) + torch.exp(self.leakyrelu(torch.mm(e87, self.a))) + torch.exp(self.leakyrelu(torch.mm(e88, self.a))) + torch.exp(self.leakyrelu(torch.mm(e89, self.a))))
            e9_e1_att = torch.exp(self.leakyrelu(torch.mm(e91, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e91, self.a))) + torch.exp(self.leakyrelu(torch.mm(e92, self.a))) + torch.exp(self.leakyrelu(torch.mm(e93, self.a))) + torch.exp(self.leakyrelu(torch.mm(e94, self.a))) + torch.exp(self.leakyrelu(torch.mm(e95, self.a))) + torch.exp(self.leakyrelu(torch.mm(e96, self.a))) + torch.exp(self.leakyrelu(torch.mm(e97, self.a))) + torch.exp(self.leakyrelu(torch.mm(e98, self.a))) + torch.exp(self.leakyrelu(torch.mm(e99, self.a))))
            e9_e2_att = torch.exp(self.leakyrelu(torch.mm(e92, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e91, self.a))) + torch.exp(self.leakyrelu(torch.mm(e92, self.a))) + torch.exp(self.leakyrelu(torch.mm(e93, self.a))) + torch.exp(self.leakyrelu(torch.mm(e94, self.a))) + torch.exp(self.leakyrelu(torch.mm(e95, self.a))) + torch.exp(self.leakyrelu(torch.mm(e96, self.a))) + torch.exp(self.leakyrelu(torch.mm(e97, self.a))) + torch.exp(self.leakyrelu(torch.mm(e98, self.a))) + torch.exp(self.leakyrelu(torch.mm(e99, self.a))))
            e9_e3_att = torch.exp(self.leakyrelu(torch.mm(e93, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e91, self.a))) + torch.exp(self.leakyrelu(torch.mm(e92, self.a))) + torch.exp(self.leakyrelu(torch.mm(e93, self.a))) + torch.exp(self.leakyrelu(torch.mm(e94, self.a))) + torch.exp(self.leakyrelu(torch.mm(e95, self.a))) + torch.exp(self.leakyrelu(torch.mm(e96, self.a))) + torch.exp(self.leakyrelu(torch.mm(e97, self.a))) + torch.exp(self.leakyrelu(torch.mm(e98, self.a))) + torch.exp(self.leakyrelu(torch.mm(e99, self.a))))
            e9_e4_att = torch.exp(self.leakyrelu(torch.mm(e94, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e91, self.a))) + torch.exp(self.leakyrelu(torch.mm(e92, self.a))) + torch.exp(self.leakyrelu(torch.mm(e93, self.a))) + torch.exp(self.leakyrelu(torch.mm(e94, self.a))) + torch.exp(self.leakyrelu(torch.mm(e95, self.a))) + torch.exp(self.leakyrelu(torch.mm(e96, self.a))) + torch.exp(self.leakyrelu(torch.mm(e97, self.a))) + torch.exp(self.leakyrelu(torch.mm(e98, self.a))) + torch.exp(self.leakyrelu(torch.mm(e99, self.a))))
            e9_e5_att = torch.exp(self.leakyrelu(torch.mm(e95, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e91, self.a))) + torch.exp(self.leakyrelu(torch.mm(e92, self.a))) + torch.exp(self.leakyrelu(torch.mm(e93, self.a))) + torch.exp(self.leakyrelu(torch.mm(e94, self.a))) + torch.exp(self.leakyrelu(torch.mm(e95, self.a))) + torch.exp(self.leakyrelu(torch.mm(e96, self.a))) + torch.exp(self.leakyrelu(torch.mm(e97, self.a))) + torch.exp(self.leakyrelu(torch.mm(e98, self.a))) + torch.exp(self.leakyrelu(torch.mm(e99, self.a))))
            e9_e6_att = torch.exp(self.leakyrelu(torch.mm(e96, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e91, self.a))) + torch.exp(self.leakyrelu(torch.mm(e92, self.a))) + torch.exp(self.leakyrelu(torch.mm(e93, self.a))) + torch.exp(self.leakyrelu(torch.mm(e94, self.a))) + torch.exp(self.leakyrelu(torch.mm(e95, self.a))) + torch.exp(self.leakyrelu(torch.mm(e96, self.a))) + torch.exp(self.leakyrelu(torch.mm(e97, self.a))) + torch.exp(self.leakyrelu(torch.mm(e98, self.a))) + torch.exp(self.leakyrelu(torch.mm(e99, self.a))))
            e9_e7_att = torch.exp(self.leakyrelu(torch.mm(e97, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e91, self.a))) + torch.exp(self.leakyrelu(torch.mm(e92, self.a))) + torch.exp(self.leakyrelu(torch.mm(e93, self.a))) + torch.exp(self.leakyrelu(torch.mm(e94, self.a))) + torch.exp(self.leakyrelu(torch.mm(e95, self.a))) + torch.exp(self.leakyrelu(torch.mm(e96, self.a))) + torch.exp(self.leakyrelu(torch.mm(e97, self.a))) + torch.exp(self.leakyrelu(torch.mm(e98, self.a))) + torch.exp(self.leakyrelu(torch.mm(e99, self.a))))
            e9_e8_att = torch.exp(self.leakyrelu(torch.mm(e98, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e91, self.a))) + torch.exp(self.leakyrelu(torch.mm(e92, self.a))) + torch.exp(self.leakyrelu(torch.mm(e93, self.a))) + torch.exp(self.leakyrelu(torch.mm(e94, self.a))) + torch.exp(self.leakyrelu(torch.mm(e95, self.a))) + torch.exp(self.leakyrelu(torch.mm(e96, self.a))) + torch.exp(self.leakyrelu(torch.mm(e97, self.a))) + torch.exp(self.leakyrelu(torch.mm(e98, self.a))) + torch.exp(self.leakyrelu(torch.mm(e99, self.a))))
            e9_e9_att = torch.exp(self.leakyrelu(torch.mm(e99, self.a))) / (torch.exp(self.leakyrelu(torch.mm(e91, self.a))) + torch.exp(self.leakyrelu(torch.mm(e92, self.a))) + torch.exp(self.leakyrelu(torch.mm(e93, self.a))) + torch.exp(self.leakyrelu(torch.mm(e94, self.a))) + torch.exp(self.leakyrelu(torch.mm(e95, self.a))) + torch.exp(self.leakyrelu(torch.mm(e96, self.a))) + torch.exp(self.leakyrelu(torch.mm(e97, self.a))) + torch.exp(self.leakyrelu(torch.mm(e98, self.a))) + torch.exp(self.leakyrelu(torch.mm(e99, self.a))))


            new_e1 = torch.mm(e1, self.W2) + torch.tanh(e11*e1_e1_att + e12*e1_e2_att + e13*e1_e3_att + e14*e1_e4_att + e15*e1_e5_att + e16*e1_e6_att + e17*e1_e7_att + e18*e1_e8_att + e19*e1_e9_att)
            new_e2 = torch.mm(e2, self.W2) + torch.tanh(e21*e2_e1_att + e22*e2_e2_att + e23*e2_e3_att + e24*e2_e4_att + e25*e2_e5_att + e26*e2_e6_att + e27*e2_e7_att + e28*e2_e8_att + e29*e2_e9_att)
            new_e3 = torch.mm(e3, self.W2) + torch.tanh(e31*e3_e1_att + e32*e3_e2_att + e33*e3_e3_att + e34*e3_e4_att + e35*e3_e5_att + e36*e3_e6_att + e37*e3_e7_att + e38*e3_e8_att + e39*e3_e9_att)
            new_e4 = torch.mm(e4, self.W2) + torch.tanh(e41*e4_e1_att + e42*e4_e2_att + e43*e4_e3_att + e44*e4_e4_att + e45*e4_e5_att + e46*e4_e6_att + e47*e4_e7_att + e48*e4_e8_att + e49*e4_e9_att)
            new_e5 = torch.mm(e5, self.W2) + torch.tanh(e51*e5_e1_att + e52*e5_e2_att + e53*e5_e3_att + e54*e5_e4_att + e55*e5_e5_att + e56*e5_e6_att + e57*e5_e7_att + e58*e5_e8_att + e59*e5_e9_att)
            new_e6 = torch.mm(e6, self.W2) + torch.tanh(e61*e6_e1_att + e62*e6_e2_att + e63*e6_e3_att + e64*e6_e4_att + e65*e6_e5_att + e66*e6_e6_att + e67*e6_e7_att + e68*e6_e8_att + e69*e6_e9_att)
            new_e7 = torch.mm(e7, self.W2) + torch.tanh(e71*e7_e1_att + e72*e7_e2_att + e73*e7_e3_att + e74*e7_e4_att + e75*e7_e5_att + e76*e7_e6_att + e77*e7_e7_att + e78*e7_e8_att + e79*e7_e9_att)
            new_e8 = torch.mm(e8, self.W2) + torch.tanh(e81*e8_e1_att + e82*e8_e2_att + e83*e8_e3_att + e84*e8_e4_att + e85*e8_e5_att + e86*e8_e6_att + e87*e8_e7_att + e88*e8_e8_att + e89*e8_e9_att)
            new_e9 = torch.mm(e9, self.W2) + torch.tanh(e91*e9_e1_att + e92*e9_e2_att + e93*e9_e3_att + e94*e9_e4_att + e95*e9_e5_att + e96*e9_e6_att + e97*e9_e7_att + e98*e9_e8_att + e99*e9_e9_att)


            re1 = self.er_pos_emb(r, e1)
            re2 = self.er_pos_emb(r, e2)
            re3 = self.er_pos_emb(r, e3)
            re4 = self.er_pos_emb(r, e4)
            re5 = self.er_pos_emb(r, e5)
            re6 = self.er_pos_emb(r, e6)
            re7 = self.er_pos_emb(r, e7)
            re8 = self.er_pos_emb(r, e8)
            re9 = self.er_pos_emb(r, e9)

            re1_att = torch.exp(torch.cosine_similarity(r, e1, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)) + torch.exp(torch.cosine_similarity(r, e9, dim=1)))
            re2_att = torch.exp(torch.cosine_similarity(r, e2, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)) + torch.exp(torch.cosine_similarity(r, e9, dim=1)))
            re3_att = torch.exp(torch.cosine_similarity(r, e3, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)) + torch.exp(torch.cosine_similarity(r, e9, dim=1)))
            re4_att = torch.exp(torch.cosine_similarity(r, e4, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)) + torch.exp(torch.cosine_similarity(r, e9, dim=1)))
            re5_att = torch.exp(torch.cosine_similarity(r, e5, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)) + torch.exp(torch.cosine_similarity(r, e9, dim=1)))
            re6_att = torch.exp(torch.cosine_similarity(r, e6, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)) + torch.exp(torch.cosine_similarity(r, e9, dim=1)))
            re7_att = torch.exp(torch.cosine_similarity(r, e7, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)) + torch.exp(torch.cosine_similarity(r, e9, dim=1)))
            re8_att = torch.exp(torch.cosine_similarity(r, e8, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)) + torch.exp(torch.cosine_similarity(r, e9, dim=1)))
            re9_att = torch.exp(torch.cosine_similarity(r, e9, dim=1)) / (torch.exp(torch.cosine_similarity(r, e1, dim=1)) + torch.exp(torch.cosine_similarity(r, e2, dim=1)) + torch.exp(torch.cosine_similarity(r, e3, dim=1)) + torch.exp(torch.cosine_similarity(r, e4, dim=1)) + torch.exp(torch.cosine_similarity(r, e5, dim=1)) + torch.exp(torch.cosine_similarity(r, e6, dim=1)) + torch.exp(torch.cosine_similarity(r, e7, dim=1)) + torch.exp(torch.cosine_similarity(r, e8, dim=1)) + torch.exp(torch.cosine_similarity(r, e9, dim=1)))

            # r = re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att + re6 * re6_att + re7 * re7_att + re8 * re8_att + re9 * re9_att
            r = torch.mm(r, self.W3) + torch.tanh(re1 * re1_att.view(-1, 1) + re2 * re2_att.view(-1, 1) + re3 * re3_att.view(-1, 1) + re4 * re4_att.view(-1, 1) + re5 * re5_att.view(-1, 1) + re6 * re6_att.view(-1, 1) + re7 * re7_att.view(-1, 1) + re8 * re8_att.view(-1, 1) + re9 * re9_att.view(-1, 1))


            e = r * new_e1 * new_e2 * new_e3 * new_e4 * new_e5 * new_e6 * new_e7 * new_e8 * new_e9


        x = e
        x = self.hidden_drop(x)
        return torch.sum(x, dim=1)

