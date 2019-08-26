import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu, projection_transH_pytorch
from jTransUP.models.transH import TransHModel
from jTransUP.models.transUP import TransUPModel

def build_model(FLAGS, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None):
    model_cls = jTransUPModel
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                user_total = user_total,
                item_total = item_total,
                entity_total = entity_total,
                relation_total = relation_total,
                i_map=i_map,
                new_map=new_map,
                isShare = FLAGS.share_embeddings,
                use_st_gumbel = FLAGS.use_st_gumbel
    )

class jTransUPModel(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                user_total,
                item_total,
                entity_total,
                relation_total,
                i_map,
                new_map,
                isShare,
                use_st_gumbel
                ):
        super(jTransUPModel, self).__init__()
        self.L1_flag = L1_flag
        self.is_share = isShare
        self.use_st_gumbel = use_st_gumbel
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
        # todo: simiplifying the init
        # transup
        user_weight = torch.FloatTensor(self.user_total, self.embedding_size)
        item_weight = torch.FloatTensor(self.item_total, self.embedding_size)
        pref_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        pref_norm_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        nn.init.xavier_uniform(user_weight)
        nn.init.xavier_uniform(item_weight)
        nn.init.xavier_uniform(pref_weight)
        nn.init.xavier_uniform(pref_norm_weight)
        # init user and item embeddings
        self.user_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.item_total, self.embedding_size)
        self.user_embeddings.weight = nn.Parameter(user_weight)
        self.item_embeddings.weight = nn.Parameter(item_weight)
        normalize_user_emb = F.normalize(self.user_embeddings.weight.data, p=2, dim=1)
        normalize_item_emb = F.normalize(self.item_embeddings.weight.data, p=2, dim=1)
        self.user_embeddings.weight.data = normalize_user_emb
        self.item_embeddings.weight.data = normalize_item_emb
        # init preference parameters
        self.pref_embeddings = nn.Embedding(self.rel_total, self.embedding_size)
        self.pref_norm_embeddings = nn.Embedding(self.rel_total, self.embedding_size)
        self.pref_embeddings.weight = nn.Parameter(pref_weight)
        self.pref_norm_embeddings.weight = nn.Parameter(pref_norm_weight)
        normalize_pref_emb = F.normalize(self.pref_embeddings.weight.data, p=2, dim=1)
        normalize_pref_norm_emb = F.normalize(self.pref_norm_embeddings.weight.data, p=2, dim=1)
        self.pref_embeddings.weight.data = normalize_pref_emb
        self.pref_norm_embeddings.weight.data = normalize_pref_norm_emb

        self.user_embeddings = to_gpu(self.user_embeddings)
        self.item_embeddings = to_gpu(self.item_embeddings)
        self.pref_embeddings = to_gpu(self.pref_embeddings)
        self.pref_norm_embeddings = to_gpu(self.pref_norm_embeddings)

        # transh
        ent_weight = torch.FloatTensor(self.ent_total-1, self.embedding_size)
        rel_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        norm_weight = torch.FloatTensor(self.rel_total, self.embedding_size)
        nn.init.xavier_uniform(ent_weight)
        nn.init.xavier_uniform(rel_weight)
        nn.init.xavier_uniform(norm_weight)
        norm_ent_weight = F.normalize(ent_weight, p=2, dim=1)
        # init user and item embeddings
        self.ent_embeddings = nn.Embedding(self.ent_total, self.embedding_size, padding_idx=self.ent_total-1)
        self.rel_embeddings = nn.Embedding(self.rel_total, self.embedding_size)
        self.norm_embeddings = nn.Embedding(self.rel_total, self.embedding_size)

        self.ent_embeddings.weight = nn.Parameter(torch.cat([norm_ent_weight, torch.zeros(1, self.embedding_size)], dim=0))
        self.rel_embeddings.weight = nn.Parameter(rel_weight)
        self.norm_embeddings.weight = nn.Parameter(norm_weight)

        normalize_rel_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.norm_embeddings.weight.data, p=2, dim=1)

        self.rel_embeddings.weight.data = normalize_rel_emb
        self.norm_embeddings.weight.data = normalize_norm_emb

        self.ent_embeddings = to_gpu(self.ent_embeddings)
        self.rel_embeddings = to_gpu(self.rel_embeddings)
        self.norm_embeddings = to_gpu(self.norm_embeddings)

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

            _, r_e, norm = self.getPreferences(u_e, ie_e, use_st_gumbel=self.use_st_gumbel)

            proj_u_e = projection_transH_pytorch(u_e, norm)
            proj_i_e = projection_transH_pytorch(ie_e, norm)

            if self.L1_flag:
                score = torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 1)
            else:
                score = torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 1)
        elif not is_rec and triples is not None:
            h, t, r = triples
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
        else:
            raise NotImplementedError
        
        return score
    
    def evaluateRec(self, u_ids, all_i_ids=None):
        batch_size = len(u_ids)
        all_i = self.item_embeddings(all_i_ids) if all_i_ids is not None and self.is_share else self.item_embeddings.weight
        item_total, dim = all_i.size()

        u = self.user_embeddings(u_ids)
        # expand u and i to pair wise match, batch * item * dim 
        u_e = u.expand(item_total, batch_size, dim).permute(1, 0, 2)
        
        i_e = all_i.expand(batch_size, item_total, dim)

        e_ids = self.paddingItems(all_i_ids.data if all_i_ids is not None else self.i_map, self.ent_total-1)
        e_var = to_gpu(V(torch.LongTensor(e_ids)))
        e_e = self.ent_embeddings(e_var).expand(batch_size, item_total, dim)
        
        ie_e = i_e + e_e

        # batch * item * dim
        _, r_e, norm = self.getPreferences(u_e, ie_e, use_st_gumbel=self.use_st_gumbel)

        proj_u_e = projection_transH_pytorch(u_e, norm)
        proj_i_e = projection_transH_pytorch(ie_e, norm)

        # batch * item
        if self.L1_flag:
            score = torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 2)
        else:
            score = torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 2)
        return score
    
    def evaluateHead(self, t, r, all_e_ids=None):
        batch_size = len(t)
        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None and self.is_share else self.ent_embeddings.weight
        ent_total, dim = all_e.size()
        # batch * dim
        t_e = self.ent_embeddings(t)
        r_e = self.rel_embeddings(r)
        norm_e = self.norm_embeddings(r)

        proj_t_e = projection_transH_pytorch(t_e, norm_e)
        c_h_e = proj_t_e - r_e
        
        # batch * entity * dim
        c_h_expand = c_h_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        norm_expand = norm_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        ent_expand = all_e.expand(batch_size, ent_total, dim)
        proj_ent_e = projection_transH_pytorch(ent_expand, norm_expand)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_h_expand-proj_ent_e), 2)
        else:
            score = torch.sum((c_h_expand-proj_ent_e) ** 2, 2)
        return score
    
    def evaluateTail(self, h, r, all_e_ids=None):
        batch_size = len(h)
        all_e = self.ent_embeddings(all_e_ids) if all_e_ids is not None and self.is_share else self.ent_embeddings.weight
        ent_total, dim = all_e.size()
        # batch * dim
        h_e = self.ent_embeddings(h)
        r_e = self.rel_embeddings(r)
        norm_e = self.norm_embeddings(r)

        proj_h_e = projection_transH_pytorch(h_e, norm_e)
        c_t_e = proj_h_e + r_e
        
        # batch * entity * dim
        c_t_expand = c_t_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)

        # batch * entity * dim
        norm_expand = norm_e.expand(ent_total, batch_size, dim).permute(1, 0, 2)
        
        ent_expand = all_e.expand(batch_size, ent_total, dim)
        proj_ent_e = projection_transH_pytorch(ent_expand, norm_expand)

        # batch * entity
        if self.L1_flag:
            score = torch.sum(torch.abs(c_t_expand-proj_ent_e), 2)
        else:
            score = torch.sum((c_t_expand-proj_ent_e) ** 2, 2)
        return score
    
    # u_e, i_e : batch * dim or batch * item * dim
    def getPreferences(self, u_e, i_e, use_st_gumbel=False):
        # use item and user embedding to compute preference distribution
        # pre_probs: batch * rel, or batch * item * rel
        pre_probs = torch.matmul(u_e + i_e, torch.t(self.pref_embeddings.weight + self.rel_embeddings.weight)) / 2
        if use_st_gumbel:
            pre_probs = self.st_gumbel_softmax(pre_probs)

        r_e = torch.matmul(pre_probs, self.pref_embeddings.weight + self.rel_embeddings.weight) / 2
        norm = torch.matmul(pre_probs, self.pref_norm_embeddings.weight + self.norm_embeddings.weight) / 2

        return pre_probs, r_e, norm
    
    # batch or batch * item
    def convert_to_one_hot(self, indices, num_classes):
        """
        Args:
            indices (Variable): A vector containing indices,
                whose size is (batch_size,).
            num_classes (Variable): The number of classes, which would be
                the second dimension of the resulting one-hot matrix.
        Returns:
            result: The one-hot matrix of size (batch_size, num_classes).
        """

        old_shape = indices.shape
        new_shape = torch.Size([i for i in old_shape] + [num_classes])
        indices = indices.unsqueeze(len(old_shape))

        one_hot = V(indices.data.new(new_shape).zero_()
                        .scatter_(len(old_shape), indices.data, 1))
        return one_hot


    def masked_softmax(self, logits):
        eps = 1e-20
        probs = F.softmax(logits, dim=len(logits.shape)-1)
        return probs

    def st_gumbel_softmax(self, logits, temperature=1.0):
        """
        Return the result of Straight-Through Gumbel-Softmax Estimation.
        It approximates the discrete sampling via Gumbel-Softmax trick
        and applies the biased ST estimator.
        In the forward propagation, it emits the discrete one-hot result,
        and in the backward propagation it approximates the categorical
        distribution via smooth Gumbel-Softmax distribution.
        Args:
            logits (Variable): A un-normalized probability values,
                which has the size (batch_size, num_classes)
            temperature (float): A temperature parameter. The higher
                the value is, the smoother the distribution is.
        Returns:
            y: The sampled output, which has the property explained above.
        """

        eps = 1e-20
        u = logits.data.new(*logits.size()).uniform_()
        gumbel_noise = V(-torch.log(-torch.log(u + eps) + eps))
        y = logits + gumbel_noise
        y = self.masked_softmax(logits=y / temperature)
        y_argmax = y.max(len(y.shape)-1)[1]
        y_hard = self.convert_to_one_hot(
            indices=y_argmax,
            num_classes=y.size(len(y.shape)-1)).float()
        y = (y_hard - y).detach() + y
        return y
    
    def reportPreference(self, u_id, i_ids):
        item_num = len(i_ids)
        # item * dim
        u_e = self.user_embeddings(u_id.expand(item_num))
        i_e = self.item_embeddings(i_ids)

        e_ids = self.paddingItems(i_ids.data, self.ent_total-1)
        e_var = to_gpu(V(torch.LongTensor(e_ids)))
        e_e = self.ent_embeddings(e_var)
        ie_e = i_e + e_e

        return self.getPreferences(u_e, ie_e, use_st_gumbel=self.use_st_gumbel)

    def disable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=False
    
    def enable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=True
