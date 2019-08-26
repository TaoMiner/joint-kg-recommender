import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu

def build_model(FLAGS, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None):
    model_cls = CFKG
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                user_total = user_total,
                item_total = item_total,
                entity_total = entity_total,
                relation_total = relation_total
    )

class CFKG(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                user_total,
                item_total,
                entity_total,
                relation_total
                ):
        super(CFKG, self).__init__()
        self.L1_flag = L1_flag
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        self.ent_total = entity_total
        # add buy relation between user and item
        self.rel_total = relation_total + 1
        self.is_pretrained = False

        # init user embeddings
        user_weight = torch.FloatTensor(self.user_total, self.embedding_size)
        nn.init.xavier_uniform(user_weight)
        
        self.user_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.user_embeddings.weight = nn.Parameter(user_weight)
        normalize_user_emb = F.normalize(self.user_embeddings.weight.data, p=2, dim=1)
        self.user_embeddings.weight.data = normalize_user_emb
        self.user_embeddings = to_gpu(self.user_embeddings)

        # init entity and relation embeddings
        ent_weight = torch.FloatTensor(self.ent_total, self.embedding_size)
        rel_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        self.ent_embeddings = nn.Embedding(self.ent_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.rel_total, self.embedding_size)

        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

        normalize_ent_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_rel_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)

        self.ent_embeddings.weight.data = normalize_ent_emb
        self.rel_embeddings.weight.data = normalize_rel_emb

        self.ent_embeddings = to_gpu(self.ent_embeddings)
        self.rel_embeddings = to_gpu(self.rel_embeddings)

        # share embedding
        self.item_embeddings = self.ent_embeddings
    
    def forward(self, ratings, triples, is_rec=True):
        
        if is_rec and ratings is not None:
            u_ids, i_ids = ratings
            batch_size = len(u_ids)

            u_e = self.user_embeddings(u_ids)
            i_e = self.item_embeddings(i_ids)

            buy_e = self.rel_embeddings(to_gpu(V(torch.LongTensor([self.rel_total-1]))))
            buy_e_expand = buy_e.expand(batch_size, self.embedding_size)
            # L1 distance
            if self.L1_flag:
                score = torch.sum(torch.abs(u_e + buy_e_expand - i_e), 1)
            # L2 distance
            else:
                score = torch.sum((u_e + buy_e_expand - i_e) ** 2, 1)

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
        all_i = self.item_embeddings(all_i_ids) if all_i_ids is not None else self.item_embeddings.weight
        item_total, dim = all_i.size()
        # batch * dim
        u_e = self.user_embeddings(u_ids)
        # batch * item * dim
        u_e_expand = u_e.expand(item_total, batch_size, dim).permute(1, 0, 2)

        buy_e = self.rel_embeddings(to_gpu(V(torch.LongTensor([self.rel_total-1]))))
        buy_e_expand = buy_e.expand(batch_size, item_total, dim)

        c_i_e = u_e_expand + buy_e_expand
        
        # batch * item * dim
        i_expand = all_i.expand(batch_size, item_total, dim)

        if self.L1_flag:
            score = torch.sum(torch.abs(c_i_e-i_expand), 2)
        else:
            score = torch.sum((c_i_e-i_expand) ** 2, 2)
        return score
    
    def evaluateHead(self, t, r, all_e_ids=None):
        batch_size = len(t)
        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None else self.ent_embeddings.weight
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
        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None else self.ent_embeddings.weight
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