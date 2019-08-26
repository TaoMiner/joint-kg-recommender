import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V

from jTransUP.utils.misc import to_gpu

def build_model(FLAGS, user_total, item_total, entity_total, relation_total, i_map=None, e_map=None, new_map=None):
    model_cls = FM
    return model_cls(
                FLAGS.embedding_size,
                user_total,
                item_total)

class FM(nn.Module):
    def __init__(self,
                embedding_size,
                user_total,
                item_total,
                ):
        super(FM, self).__init__()
        self.embedding_size = embedding_size
        self.user_total = user_total
        self.item_total = item_total
        self.is_pretrained = False

        # init user and item embeddings
        user_weight = torch.FloatTensor(self.user_total, self.embedding_size)
        item_weight = torch.FloatTensor(self.item_total, self.embedding_size)
        user_bias = torch.FloatTensor(self.user_total)
        item_bias = torch.FloatTensor(self.item_total)
        nn.init.xavier_uniform(user_weight)
        nn.init.xavier_uniform(item_weight)
        nn.init.constant(user_bias, 0)
        nn.init.constant(item_bias, 0)
        # init user and item embeddings
        self.user_embeddings = nn.Embedding(self.user_total, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.item_total, self.embedding_size)
        self.user_bias = nn.Embedding(self.user_total, 1)
        self.item_bias = nn.Embedding(self.item_total, 1)
        self.user_embeddings.weight = nn.Parameter(user_weight)
        self.item_embeddings.weight = nn.Parameter(item_weight)
        self.user_bias.weight = nn.Parameter(user_bias, 1)
        self.item_bias.weight = nn.Parameter(item_bias, 1)
        normalize_user_emb = F.normalize(self.user_embeddings.weight.data, p=2, dim=1)
        normalize_item_emb = F.normalize(self.item_embeddings.weight.data, p=2, dim=1)
        self.user_embeddings.weight.data = normalize_user_emb
        self.item_embeddings.weight.data = normalize_item_emb

        self.user_embeddings = to_gpu(self.user_embeddings)
        self.item_embeddings = to_gpu(self.item_embeddings)
        self.user_bias = to_gpu(self.user_bias)
        self.item_bias = to_gpu(self.item_bias)

        self.bias = nn.Parameter(to_gpu(torch.FloatTensor([0.0])))

    def forward(self, u_ids, i_ids):
        batch_size = len(u_ids)

        u_e = self.user_embeddings(u_ids)
        i_e = self.item_embeddings(i_ids)
        u_b = self.user_bias(u_ids).squeeze()
        i_b = self.item_bias(i_ids).squeeze()

        y = self.bias.expand(batch_size) + u_b + i_b + torch.bmm(u_e.unsqueeze(1), i_e.unsqueeze(2)).squeeze()
        return y
    
    def evaluate(self, u_ids):
        batch_size = len(u_ids)
        u_e = self.user_embeddings(u_ids)
        u_b = self.user_bias(u_ids).squeeze()

        # expand to batch * item
        u_b_e = u_b.expand(self.item_total, batch_size).permute(1, 0)
        i_b_e = self.item_bias.weight.squeeze().expand(batch_size, self.item_total)

        y_e = self.bias.expand(batch_size, self.item_total) + u_b_e + i_b_e + torch.matmul(u_e, self.item_embeddings.weight.t())

        return y_e
    
    def disable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=False
    
    def enable_grad(self):
        for name, param in self.named_parameters():
            param.requires_grad=True