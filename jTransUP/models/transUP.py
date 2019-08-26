import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu, projection_transH_pytorch

def build_model(FLAGS, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None):
    model_cls = TransUPModel
    return model_cls(
                L1_flag = FLAGS.L1_flag,
                embedding_size = FLAGS.embedding_size,
                user_total = user_total,
                item_total = item_total,
                preference_total = FLAGS.num_preferences,
                use_st_gumbel = FLAGS.use_st_gumbel
    )

class TransUPModel(nn.Module):
    def __init__(self,
                L1_flag,
                embedding_size,
                user_total,
                item_total,
                preference_total,
                use_st_gumbel
                ):
        super(TransUPModel, self).__init__()
        self.L1_flag = L1_flag
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        self.preference_total = preference_total
        self.is_pretrained = False
        self.use_st_gumbel = use_st_gumbel

        user_weight = torch.FloatTensor(self.user_total, self.embedding_size)
        item_weight = torch.FloatTensor(self.item_total, self.embedding_size)
        pref_weight = torch.FloatTensor(self.preference_total, self.embedding_size)
        norm_weight = torch.FloatTensor(self.preference_total, self.embedding_size)
        nn.init.xavier_uniform(user_weight)
        nn.init.xavier_uniform(item_weight)
        nn.init.xavier_uniform(pref_weight)
        nn.init.xavier_uniform(norm_weight)
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
        self.pref_embeddings = nn.Embedding(self.preference_total, self.embedding_size)
        self.pref_norm_embeddings = nn.Embedding(self.preference_total, self.embedding_size)
        self.pref_embeddings.weight = nn.Parameter(pref_weight)
        self.pref_norm_embeddings.weight = nn.Parameter(norm_weight)
        normalize_pref_emb = F.normalize(self.pref_embeddings.weight.data, p=2, dim=1)
        normalize_norm_emb = F.normalize(self.pref_norm_embeddings.weight.data, p=2, dim=1)
        self.pref_embeddings.weight.data = normalize_pref_emb
        self.pref_norm_embeddings.weight.data = normalize_norm_emb

        self.user_embeddings = to_gpu(self.user_embeddings)
        self.item_embeddings = to_gpu(self.item_embeddings)
        self.pref_embeddings = to_gpu(self.pref_embeddings)
        self.pref_norm_embeddings = to_gpu(self.pref_norm_embeddings)

    def forward(self, u_ids, i_ids):
        u_e = self.user_embeddings(u_ids)
        i_e = self.item_embeddings(i_ids)
        
        _, r_e, norm = self.getPreferences(u_e, i_e, use_st_gumbel=self.use_st_gumbel)

        proj_u_e = projection_transH_pytorch(u_e, norm)
        proj_i_e = projection_transH_pytorch(i_e, norm)

        if self.L1_flag:
            score = torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 1)
        else:
            score = torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 1)
        return score
    
    def evaluate(self, u_ids):
        batch_size = len(u_ids)
        u = self.user_embeddings(u_ids)
        # expand u and i to pair wise match, batch * item * dim 
        u_e = u.expand(self.item_total, batch_size, self.embedding_size).permute(1, 0, 2)
        i_e = self.item_embeddings.weight.expand(batch_size, self.item_total, self.embedding_size)

        # batch * item * dim
        _, r_e, norm = self.getPreferences(u_e, i_e, use_st_gumbel=self.use_st_gumbel)

        proj_u_e = projection_transH_pytorch(u_e, norm)
        proj_i_e = projection_transH_pytorch(i_e, norm)

        # batch * item
        if self.L1_flag:
            score = torch.sum(torch.abs(proj_u_e + r_e - proj_i_e), 2)
        else:
            score = torch.sum((proj_u_e + r_e - proj_i_e) ** 2, 2)
        return score
    
    # u_e, i_e : batch * dim or batch * item * dim
    def getPreferences(self, u_e, i_e, use_st_gumbel=False):
        # use item and user embedding to compute preference distribution
        # pre_probs: batch * rel, or batch * item * rel
        pre_probs = torch.matmul(u_e + i_e, torch.t(self.pref_embeddings.weight)) / 2
        if use_st_gumbel:
            pre_probs = self.st_gumbel_softmax(pre_probs)

        r_e = torch.matmul(pre_probs, self.pref_embeddings.weight)
        norm = torch.matmul(pre_probs, self.pref_norm_embeddings.weight)

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
        # item_num * dim
        u_e = self.user_embeddings(u_id.expand(item_num))
        # item_num * dim
        i_e = self.item_embeddings(i_ids)
        # item_num * relation_total

        return self.getPreferences(u_e, i_e, use_st_gumbel=self.use_st_gumbel)
    
    def disable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=False
    
    def enable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=True