import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from jTransUP.utils.misc import to_gpu, projection_transR_pytorch, projection_transR_pytorch_batch

def build_model(FLAGS, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None):
    model_cls = CKE
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                user_total = user_total,
                item_total = item_total,
                entity_total = entity_total,
                relation_total = relation_total,
                i_map=i_map,
                new_map=new_map
    )

class CKE(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                user_total,
                item_total,
                entity_total,
                relation_total,
                i_map,
                new_map
                ):
        super(CKE, self).__init__()
        self.L1_flag = L1_flag
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        # padding when item are not aligned with any entity
        self.ent_total = entity_total + 1
        self.rel_total = relation_total
        self.is_pretrained = False
        # store item to item-entity dic
        self.i_map = i_map
        # store item-entity to (entity, item)
        self.new_map = new_map

        # bprmf
        # init user and item embeddings
        user_weight = torch.FloatTensor(self.user_total, self.embedding_size)
        item_weight = torch.FloatTensor(self.item_total, self.embedding_size)
        nn.init.xavier_uniform(user_weight)
        nn.init.xavier_uniform(item_weight)
        self.user_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.item_total, self.embedding_size)
        self.user_embeddings.weight = nn.Parameter(user_weight)
        self.item_embeddings.weight = nn.Parameter(item_weight)
        normalize_user_emb = F.normalize(self.user_embeddings.weight.data, p=2, dim=1)
        normalize_item_emb = F.normalize(self.item_embeddings.weight.data, p=2, dim=1)
        self.user_embeddings.weight.data = normalize_user_emb
        self.item_embeddings.weight.data = normalize_item_emb

        self.user_embeddings = to_gpu(self.user_embeddings)
        self.item_embeddings = to_gpu(self.item_embeddings)

        # transR
        
        ent_weight = torch.FloatTensor(self.ent_total-1, self.embedding_size)
        rel_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        proj_weight = torch.FloatTensor(self.rel_total, self.embedding_size * self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)

        norm_ent_weight = F.normalize(ent_weight, p=2, dim=1)
        
        if self.is_pretrained:
            nn.init.eye(proj_weight)
            proj_weight = proj_weight.view(-1).expand(self.relation_total, -1)
        else:
            nn.init.xavier_uniform(proj_weight)
            
        # init user and item embeddings
        self.ent_embeddings = nn.Embedding(self.ent_total, self.embedding_size, padding_idx=self.ent_total-1)

        self.rel_embeddings = nn.Embedding(self.rel_total, self.embedding_size)
        self.proj_embeddings = nn.Embedding(self.rel_total, self.embedding_size * self.embedding_size)

        self.ent_embeddings.weight = nn.Parameter(torch.cat([norm_ent_weight, torch.zeros(1, self.embedding_size)], dim=0))
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.proj_embeddings.weight = nn.Parameter(proj_weight)

        # normalize_ent_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_rel_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        # normalize_proj_emb = F.normalize(self.proj_embeddings.weight.data, p=2, dim=1)

        # self.ent_embeddings.weight.data = normalize_ent_emb
        self.rel_embeddings.weight.data = normalize_rel_emb
        # self.proj_embeddings.weight.data = normalize_proj_emb

        self.ent_embeddings = to_gpu(self.ent_embeddings)
        self.rel_embeddings = to_gpu(self.rel_embeddings)
        self.proj_embeddings = to_gpu(self.proj_embeddings)
    
    def paddingItems(self, i_ids, pad_index):
        padded_e_ids = []
        for i_id in i_ids:
            new_index = self.i_map[i_id]
            ent_id = self.new_map[new_index][0]
            padded_e_ids.append(ent_id if ent_id != -1 else pad_index)
        return padded_e_ids

    def forward(self, ratings, triples, is_rec=True):
        
        if is_rec and ratings is not None:
            u_ids, i_ids = ratings

            e_ids = self.paddingItems(i_ids.data, self.ent_total-1)
            e_var = to_gpu(V(torch.LongTensor(e_ids)))

            u_e = self.user_embeddings(u_ids)
            i_e = self.item_embeddings(i_ids)
            e_e = self.ent_embeddings(e_var)
            ie_e = i_e + e_e

            score = torch.bmm(u_e.unsqueeze(1), ie_e.unsqueeze(2)).squeeze()
        elif not is_rec and triples is not None:
            h, t, r = triples
            h_e = self.ent_embeddings(h)
            t_e = self.ent_embeddings(t)
            r_e = self.rel_embeddings(r)
            proj_e = self.proj_embeddings(r)

            proj_h_e = projection_transR_pytorch(h_e, proj_e)
            proj_t_e = projection_transR_pytorch(t_e, proj_e)

            if self.L1_flag:
                score = torch.sum(torch.abs(proj_h_e + r_e - proj_t_e), 1)
            else:
                score = torch.sum((proj_h_e + r_e - proj_t_e) ** 2, 1)
        else:
            raise NotImplementedError
        
        return score
    
    def evaluateRec(self, u_ids, all_i_ids=None):
        batch_size = len(u_ids)
        i_ids = range(len(self.item_embeddings.weight))
        e_ids = self.paddingItems(i_ids, self.ent_total-1)
        e_var = to_gpu(V(torch.LongTensor(e_ids)))
        e_e = self.ent_embeddings(e_var)

        all_ie_e = self.item_embeddings.weight + e_e

        u_e = self.user_embeddings(u_ids)

        return torch.matmul(u_e, all_ie_e.t())
    
    def evaluateHead(self, t, r, all_e_ids=None):
        batch_size = len(t)
        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None and self.is_share else self.ent_embeddings.weight
        ent_total, dim = all_e.size()
        # batch * dim
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)
        # batch* dim*dim
        proj_e = self.proj_embeddings(r)
        # batch * dim
        proj_t_e = projection_transR_pytorch(t_e, proj_e)
        c_h_e = proj_t_e - r_e
        
        # batch * entity * dim
        c_h_expand = c_h_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        proj_ent_expand = projection_transR_pytorch_batch(all_e, proj_e)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_h_expand-proj_ent_expand), 2)
        else:
            score = torch.sum((c_h_expand-proj_ent_expand) ** 2, 2)
        return score
    
    def evaluateTail(self, h, r, all_e_ids=None):
        batch_size = len(h)
        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None and self.is_share else self.ent_embeddings.weight
        ent_total, dim = all_e.size()
        # batch * dim
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        # batch* dim*dim
        proj_e = self.proj_embeddings(r)
        # batch * dim
        proj_h_e = projection_transR_pytorch(h_e, proj_e)
        c_t_e = proj_h_e + r_e
        
        # batch * entity * dim
        c_t_expand = c_t_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        proj_ent_expand = projection_transR_pytorch_batch(all_e, proj_e)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_t_expand-proj_ent_expand), 2)
        else:
            score = torch.sum((c_t_expand-proj_ent_expand) ** 2, 2)
        return score

    def disable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=False
    
    def enable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=True