import torch
from collections import deque
import numpy as np
from jTransUP.utils.evaluation import ndcg_at_k
import heapq
import time
from itertools import groupby
import multiprocessing
import math

USE_CUDA = torch.cuda.is_available()

def to_gpu(var):
    if USE_CUDA:
        return var.cuda()
    return var

def projection_transH_pytorch(original, norm):
    return original - torch.sum(original * norm, dim=len(original.size())-1, keepdim=True) * norm

def projection_transR_pytorch(original, proj_matrix):
    ent_embedding_size = original.shape[1]
    rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size
    original = original.view(-1, ent_embedding_size, 1)
    proj_matrix = proj_matrix.view(-1, rel_embedding_size, ent_embedding_size)
    return torch.matmul(proj_matrix, original).view(-1, rel_embedding_size)

# original: E*d2, proj: b*d1*d2
def projection_transR_pytorch_batch(original, proj_matrix):
    ent_embedding_size = original.shape[1]
    rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size
    proj_matrix = proj_matrix.view(-1, rel_embedding_size, ent_embedding_size)
    return torch.matmul(proj_matrix, original.transpose(0,1)).transpose(1,2)

# batch * dim
def projection_transD_pytorch_samesize(entity_embedding, entity_projection, relation_projection):
    return entity_embedding + torch.sum(entity_embedding * entity_projection, dim=len(entity_embedding.size())-1, keepdim=True) * relation_projection

class Accumulator(object):
    """Accumulator. Makes it easy to keep a trailing list of statistics."""

    def __init__(self, maxlen=None):
        self.maxlen = maxlen
        self.cache = dict()

    def add(self, key, val):
        self.cache.setdefault(key, deque(maxlen=self.maxlen)).append(val)

    def get(self, key, clear=True):
        ret = self.cache.get(key, [])
        if clear:
            try:
                del self.cache[key]
            except BaseException:
                pass
        return ret

    def get_avg(self, key, clear=True):
        return np.array(self.get(key, clear)).mean()

class MyEvalKGProcess(multiprocessing.Process):
    def __init__(self, L, eval_dict, all_dicts=None, descending=True, topn=10, queue=None):
        super(MyEvalKGProcess, self).__init__()
        self.queue = queue
        self.L = L
        self.eval_dict = eval_dict
        self.all_dicts = all_dicts
        self.topn = topn
        self.descending = descending

    def run(self):
        while True:
            pred_scores = self.queue.get()
            try:
                self.process_data(pred_scores, self.eval_dict, all_dicts=self.all_dicts)
            except:
                time.sleep(5)
                self.process_data(pred_scores, self.eval_dict, all_dicts=self.all_dicts)
            self.queue.task_done()

    def process_data(self, pred_scores, eval_dict, all_dicts=None):
        for pred in pred_scores:
            if pred[0] not in eval_dict: continue
            gold = eval_dict[pred[0]]
            # ids to be filtered
            fliter_samples = None
            if all_dicts is not None:
                fliter_samples = set()
                for dic in all_dicts:
                    if pred[0] in dic:
                        fliter_samples.update(dic[pred[0]])

            per_scores = pred[1] if not self.descending else -pred[1]
            hits, gold_ranks, gold_ids = getKGPerformance(per_scores, gold, fliter_samples=fliter_samples, topn=self.topn)
            self.L.extend( list(zip(hits, gold_ranks, [pred[0]]*len(hits), gold_ids)) )

# pred_scores: batch * item, [(id, numpy.array), ...], all_dicts:(train_dict, valid_dict, test_dict)
def evalKGProcess(pred_scores, eval_dict, all_dicts=None, descending=True, num_processes=multiprocessing.cpu_count(), topn=10, queue_limit=10):
    offset = math.ceil(float(len(pred_scores)) / queue_limit)
    grouped_lists = [pred_scores[i:i+offset] for i in range(0,len(pred_scores),offset)]

    with multiprocessing.Manager() as manager:
        L = manager.list()
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = MyEvalKGProcess(L, eval_dict, all_dicts=all_dicts, descending=descending, topn=topn, queue=queue)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for sub_list in grouped_lists:
            if len(sub_list) == 0 : continue
            queue.put(sub_list)
        queue.join()

        results = list(L)

        for worker in workerList:
            worker.terminate()

    return results

# pred: numpy.array, gold,filter: set()
def getKGPerformance(pred, gold, fliter_samples=None, topn=10):
    # index of pred is also ids, id's rank
    pred_ranks = np.argsort(pred)
    
    gold_ranks = []
    hits = []
    gold_ids = []
    current_rank = 0
    topn_to_skip = 0
    for rank_id in pred_ranks:
        if fliter_samples is not None and rank_id in fliter_samples :
            if current_rank < topn : topn_to_skip += 1
            continue
        if rank_id in gold:
            gold_ranks.append(current_rank)
            gold_ids.append(rank_id)
            hits.append(1 if current_rank < topn else 0)
            if len(gold_ranks) == len(gold) : break
        else:
            current_rank += 1

    return hits, gold_ranks, gold_ids

class MyEvalRecProcess(multiprocessing.Process):
    def __init__(self, L, eval_dict, all_dicts=None, descending=True, topn=10, queue=None):
        super(MyEvalRecProcess, self).__init__()
        self.queue = queue
        self.L = L
        self.eval_dict = eval_dict
        self.all_dicts = all_dicts
        self.topn = topn
        self.descending = descending

    def run(self):
        while True:
            pred_scores = self.queue.get()
            try:
                self.process_data(pred_scores, self.eval_dict, all_dicts=self.all_dicts)
            except:
                time.sleep(5)
                self.process_data(pred_scores, self.eval_dict, all_dicts=self.all_dicts)
            self.queue.task_done()

    def process_data(self, pred_scores, eval_dict, all_dicts=None):
        for pred in pred_scores:
            if pred[0] not in eval_dict: continue
            gold = eval_dict[pred[0]]
            # ids to be filtered
            fliter_samples = None
            if all_dicts is not None:
                fliter_samples = set()
                for dic in all_dicts:
                    if pred[0] in dic:
                        fliter_samples.update(dic[pred[0]])

            per_scores = pred[1] if not self.descending else -pred[1]
            f1, p, r, hit, ndcg, top_ids = getRecPerformance(per_scores, gold, fliter_samples=fliter_samples, topn=self.topn)

            self.L.append( [f1, p, r, hit, ndcg, (pred[0], top_ids, gold)] )

# pred_scores: batch * item, [(id, numpy.array), ...], all_dicts:(train_dict, valid_dict, test_dict)
def evalRecProcess(pred_scores, eval_dict, all_dicts=None, descending=True, num_processes=multiprocessing.cpu_count(), topn=10, queue_limit=10):
    offset = math.ceil(float(len(pred_scores)) / queue_limit)
    grouped_lists = [pred_scores[i:i+offset] for i in range(0,len(pred_scores),offset)]

    with multiprocessing.Manager() as manager:
        L = manager.list()
        queue = multiprocessing.JoinableQueue()
        workerList = []
        for i in range(num_processes):
            worker = MyEvalRecProcess(L, eval_dict, all_dicts=all_dicts, descending=descending, topn=topn, queue=queue)
            workerList.append(worker)
            worker.daemon = True
            worker.start()

        for sub_list in grouped_lists:
            if len(sub_list) == 0 : continue
            queue.put(sub_list)
        queue.join()

        results = list(L)

        for worker in workerList:
            worker.terminate()

    return results

# pred: numpy.array, gold,filter: set()
def getRecPerformance(pred, gold, fliter_samples=None, topn=10):
    # index of pred is also ids
    pred_ranks = np.argsort(pred)

    hits = []
    current_rank = 0
    topn_to_skip = 0
    top_ids = []
    for rank_id in pred_ranks:
        if fliter_samples is not None and rank_id in fliter_samples :
            if current_rank < topn : topn_to_skip += 1
            continue

        hits.append(1 if rank_id in gold else 0)
        top_ids.append(rank_id)
        current_rank += 1
        if current_rank >= topn : break

    # hit number, how many preds in gold
    hits_count = sum(hits)

    k = len(hits)
    k_gold = len(gold)
    f1 = 0.0
    p = 0.0
    r = 0.0
    ndcg = 0.0
    hit = 1 if hits_count > 0 else 0

    if hits_count > 0:
        p = float(hits_count) / k
        r = float(hits_count) / k_gold
        f1 = 2 * p * r / (p + r)
        ndcg = ndcg_at_k(hits, k)

    return f1, p, r, hit, ndcg, top_ids

def recursively_set_device(inp, gpu=USE_CUDA):
    if hasattr(inp, 'keys'):
        for k in list(inp.keys()):
            inp[k] = recursively_set_device(inp[k], USE_CUDA)
    elif isinstance(inp, list):
        return [recursively_set_device(ii, USE_CUDA) for ii in inp]
    elif isinstance(inp, tuple):
        return (recursively_set_device(ii, USE_CUDA) for ii in inp)
    elif hasattr(inp, 'cpu'):
        if USE_CUDA:
            inp = inp.cuda()
        else:
            inp = inp.cpu()
    return inp