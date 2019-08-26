import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu, projection_transH_pytorch
from jTransUP.models.transH import TransHModel
from jTransUP.models.transUP import TransUPModel

def build_model(FLAGS, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None):
    model_cls = coFM
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                user_total = user_total,
                item_total = item_total,
                entity_total = entity_total,
                relation_total = relation_total,
                isShare = FLAGS.share_embeddings
    )

class coFM(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                user_total,
                item_total,
                entity_total,
                relation_total,
                isShare
                ):
        super(coFM, self).__init__()
        self.L1_flag = L1_flag
        self.is_share = isShare
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        self.ent_total = entity_total
        self.rel_total = relation_total
        self.is_pretrained = False
        # fm
        user_weight = torch.FloatTensor(self.user_total, self.embedding_size)
        nn.init.xavier_uniform(user_weight)
        self.user_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.user_embeddings.weight = nn.Parameter(user_weight)
        normalize_user_emb = F.normalize(self.user_embeddings.weight.data, p=2, dim=1)
        self.user_embeddings.weight.data = normalize_user_emb
        self.user_embeddings = to_gpu(self.user_embeddings)

        user_bias = torch.FloatTensor(self.user_total)
        item_bias = torch.FloatTensor(self.item_total)
        nn.init.constant(user_bias, 0)
        nn.init.constant(item_bias, 0)
        self.user_bias = nn.Embedding(self.user_total, 1)
        self.item_bias = nn.Embedding(self.item_total, 1)
        self.user_bias.weight = nn.Parameter(user_bias, 1)
        self.item_bias.weight = nn.Parameter(item_bias, 1)

        self.user_bias = to_gpu(self.user_bias)
        self.item_bias = to_gpu(self.item_bias)

        self.bias = nn.Parameter(to_gpu(torch.FloatTensor([0.0])))

        # trane
        
        rel_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        nn.init.xavier_uniform(rel_weight)
        self.rel_embeddings = nn.Embedding(self.rel_total, self.embedding_size)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        normalize_rel_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.rel_embeddings.weight.data = normalize_rel_emb
        self.rel_embeddings = to_gpu(self.rel_embeddings)

        # shared embedding
        ent_weight = torch.FloatTensor(self.ent_total, self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        self.ent_embeddings = nn.Embedding(self.ent_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        normalize_ent_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_ent_emb
        self.ent_embeddings = to_gpu(self.ent_embeddings)

        if self.is_share:
            assert self.item_total == self.ent_total, "item numbers didn't match entities!"
            self.item_embeddings = self.ent_embeddings
        else:
            item_weight = torch.FloatTensor(self.item_total, self.embedding_size)
            nn.init.xavier_uniform(item_weight)
            self.item_embeddings = nn.Embedding(self.item_total, self.embedding_size)
            self.item_embeddings.weight = nn.Parameter(item_weight)
            normalize_item_emb = F.normalize(self.item_embeddings.weight.data, p=2, dim=1)
            self.item_embeddings.weight.data = normalize_item_emb
            self.item_embeddings = to_gpu(self.item_embeddings)

    def forward(self, ratings, triples, is_rec=True):
        
        if is_rec and ratings is not None:
            u_ids, i_ids = ratings
            batch_size = len(u_ids)

            u_e = self.user_embeddings(u_ids)
            i_e = self.item_embeddings(i_ids)
            u_b = self.user_bias(u_ids).squeeze()
            i_b = self.item_bias(i_ids).squeeze()

            score = self.bias.expand(batch_size) + u_b + i_b + torch.bmm(u_e.unsqueeze(1), i_e.unsqueeze(2)).squeeze()

        elif not is_rec and triples is not None:
            h, t, r = triples
            h_e = self.ent_embeddings(h)
            t_e = self.ent_embeddings(t)
            r_e = self.rel_embeddings(r)
            
            # L1 distance
            if self.L1_flag:
                score = torch.sum(torch.abs(h_e + r_e - t_e), 1)
            # L2 distance
            else:
                score = torch.sum((h_e + r_e - t_e) ** 2, 1)
        else:
            raise NotImplementedError
        
        return score
    
    def evaluateRec(self, u_ids, all_i_ids=None):
        batch_size = len(u_ids)
        all_i = self.item_embeddings(all_i_ids) if all_i_ids is not None and self.is_share else self.item_embeddings.weight
        all_i_b = self.item_bias(all_i_ids) if all_i_ids is not None and self.is_share else self.item_bias.weight
        item_total, _ = all_i.size()

        u_e = self.user_embeddings(u_ids)
        u_b = self.user_bias(u_ids).squeeze()
        # expand to batch * item
        u_b_e = u_b.expand(item_total, batch_size).permute(1, 0)
        i_b_e = all_i_b.squeeze().expand(batch_size, item_total)

        score = self.bias.expand(batch_size, item_total) + u_b_e + i_b_e + torch.matmul(u_e, all_i.t())

        return score
    
    def evaluateHead(self, t, r, all_e_ids=None):
        batch_size = len(t)

        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None and self.is_share else self.ent_embeddings.weight
        ent_total, dim = all_e.size()

        # batch * dim
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)

        c_h_e = t_e - r_e
        
        # batch * entity * dim
        c_h_expand = c_h_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        ent_expand = all_e.expand(batch_size, ent_total, dim)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_h_expand-ent_expand), 2)
        else:
            score = torch.sum((c_h_expand-ent_expand) ** 2, 2)

        return score
    
    def evaluateTail(self, h, r, all_e_ids=None):
        batch_size = len(h)

        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None and self.is_share else self.ent_embeddings.weight
        ent_total, dim = all_e.size()

        # batch * dim
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)

        c_t_e = h_e + r_e
        
        # batch * entity * dim
        c_t_expand = c_t_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        ent_expand = all_e.expand(batch_size, ent_total, dim)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_t_expand-ent_expand), 2)
        else:
            score = torch.sum((c_t_expand-ent_expand) ** 2, 2)
        return score
    
    def disable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=False
    
    def enable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=True