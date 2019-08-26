import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu, projection_transH_pytorch

def build_model(FLAGS, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None):
    model_cls = TransHModel
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                ent_total = entity_total,
                rel_total = relation_total
    )

class TransHModel(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                ent_total,
                rel_total
                ):
        super(TransHModel, self).__init__()
        self.L1_flag = L1_flag
        self.embedding_size = embedding_size
        self.ent_total = ent_total
        self.rel_total = rel_total
        self.is_pretrained = False

        ent_weight = torch.FloatTensor(self.ent_total, self.embedding_size)
        rel_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        norm_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        nn.init.xavier_uniform(norm_weight)
        # init user and item embeddings
        self.ent_embeddings = nn.Embedding(self.ent_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.rel_total, self.embedding_size)
        self.norm_embeddings = nn.Embedding(self.rel_total, self.embedding_size)

        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.norm_embeddings.weight = nn.Parameter(norm_weight)

        normalize_ent_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_rel_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.norm_embeddings.weight.data, p=2, dim=1)

        self.ent_embeddings.weight.data = normalize_ent_emb
        self.rel_embeddings.weight.data = normalize_rel_emb
        self.norm_embeddings.weight.data = normalize_norm_emb

        self.ent_embeddings = to_gpu(self.ent_embeddings)
        self.rel_embeddings = to_gpu(self.rel_embeddings)
        self.norm_embeddings = to_gpu(self.norm_embeddings)

    def forward(self, h, t, r):
        h_e = self.ent_embeddings(h)
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)
        norm_e = self.norm_embeddings(r)

        proj_h_e = projection_transH_pytorch(h_e, norm_e)
        proj_t_e = projection_transH_pytorch(t_e, norm_e)

        if self.L1_flag:
            score = torch.sum(torch.abs(proj_h_e + r_e - proj_t_e), 1)
        else:
            score = torch.sum((proj_h_e + r_e - proj_t_e) ** 2, 1)
        return score
    
    def evaluateHead(self, t, r):
        batch_size = len(t)
        # batch * dim
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)
        norm_e = self.norm_embeddings(r)

        proj_t_e = projection_transH_pytorch(t_e, norm_e)
        c_h_e = proj_t_e - r_e
        
        # batch * entity * dim
        c_h_expand = c_h_e.expand(self.ent_total, batch_size, self.embedding_size).permute(1, 0, 2)

        # batch * entity * dim
        norm_expand = norm_e.expand(self.ent_total, batch_size, self.embedding_size).permute(1, 0, 2)
        ent_expand = self.ent_embeddings.weight.expand(batch_size, self.ent_total, self.embedding_size)
        proj_ent_e = projection_transH_pytorch(ent_expand, norm_expand)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_h_expand-proj_ent_e), 2)
        else:
            score = torch.sum((c_h_expand-proj_ent_e) ** 2, 2)
        return score
    
    def evaluateTail(self, h, r):
        batch_size = len(h)
        # batch * dim
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        norm_e = self.norm_embeddings(r)

        proj_h_e = projection_transH_pytorch(h_e, norm_e)
        c_t_e = proj_h_e + r_e
        
        # batch * entity * dim
        c_t_expand = c_t_e.expand(self.ent_total, batch_size, self.embedding_size).permute(1, 0, 2)

        # batch * entity * dim
        norm_expand = norm_e.expand(self.ent_total, batch_size, self.embedding_size).permute(1, 0, 2)
        ent_expand = self.ent_embeddings.weight.expand(batch_size, self.ent_total, self.embedding_size)
        proj_ent_e = projection_transH_pytorch(ent_expand, norm_expand)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_t_expand-proj_ent_e), 2)
        else:
            score = torch.sum((c_t_expand-proj_ent_e) ** 2, 2)
        return score
    
    def disable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=False
    
    def enable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=True