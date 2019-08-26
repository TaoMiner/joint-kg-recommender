import math
import heapq
import numpy as np


def eval_model_pro(y_gnd, y_pre, K, row_len):
    mat_gnd = np.reshape(y_gnd, (-1, row_len))
    mat_pre = np.reshape(y_pre, (-1, row_len))

    hits, ndcgs= eval_model(mat_gnd, mat_pre, K)
    return hits, ndcgs


def eval_model(y_gnd, y_pre, K):
    ndcgs, hits = [], []
    y_gnd = y_gnd.tolist()
    y_pre = y_pre.tolist()

    for i, i_gnd in enumerate(y_gnd):
        i_pre = y_pre[i]
        hit, ndcg = eval_one_rating(i_gnd, i_pre, K)
        hits.append(hit)
        ndcgs.append(ndcg)

    hits = np.array(hits).mean()
    ndcgs = np.array(ndcgs).mean()

    return hits, ndcgs


def eval_one_rating(i_gnd, i_pre, K):
    if sum(i_pre) == 0:
        return 0, 0
    map_score = {}
    for item, score in enumerate(i_pre):
        map_score[item] = score

    rank_list = heapq.nlargest(K, map_score, key=map_score.get) #return rank index 
    target_item = i_gnd.index(1)

    hit = get_hit_ratio(rank_list, target_item)
    ndcg = get_ndcg(rank_list, target_item)

    return hit, ndcg


def get_hit_ratio(rank_list, target_item):
    for item in rank_list:
        if item == target_item:
            return 1
    return 0


def get_ndcg(rank_list, target_item):
    for i, item in enumerate(rank_list):
        if item == target_item:
            return math.log(2) / math.log(i + 2)
    return 0
