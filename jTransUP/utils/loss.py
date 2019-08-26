import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from jTransUP.utils.misc import to_gpu

class marginLoss(nn.Module):
    def __init__(self):
        super(marginLoss, self).__init__()

    def forward(self, pos, neg, margin):
        zero_tensor = to_gpu(torch.FloatTensor(pos.size()))
        zero_tensor.zero_()
        zero_tensor = autograd.Variable(zero_tensor)
        return torch.sum(torch.max(pos - neg + margin, zero_tensor))

def orthogonalLoss(rel_embeddings, norm_embeddings):
    return torch.sum(torch.sum(norm_embeddings * rel_embeddings, dim=1, keepdim=True) ** 2 / torch.sum(rel_embeddings ** 2, dim=1, keepdim=True))

def normLoss(embeddings, dim=1):
    norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
    return torch.sum(torch.max(norm - to_gpu(autograd.Variable(torch.FloatTensor([1.0]))), to_gpu(autograd.Variable(torch.FloatTensor([0.0])))))
'''
def normLoss(embeddings, dim=1):
    norm = torch.sum(embeddings ** 2, dim=dim, keepdim=True)
    return torch.sum(torch.max(norm - 1.0, to_gpu(autograd.Variable(torch.FloatTensor([0.0])))))
'''
def bprLoss(pos, neg, target=1.0):
    loss = - F.logsigmoid(target * ( pos - neg ))
    return loss.mean()

def pNormLoss(emb1, emb2, L1_flag=False):
    if L1_flag:
        distance = torch.sum(torch.abs(emb1 - emb2), 1)
    else:
        distance = torch.sum((emb1 - emb2) ** 2, 1)
    return distance.mean()