import random
from copy import deepcopy
import numpy as np

def getTrainRatingBatch(rating_batch, item_total, all_dicts=None):
    u_ids = [rating.u for rating in rating_batch]
    pi_ids = [rating.i for rating in rating_batch]
    # yield u, pi, ni, each list contains batch size ids,
    u, pi, ni = addNegRatings(rating_batch.tolist(), item_total, all_dicts=all_dicts)
    return u, pi, ni

def getTrainTripleBatch(triple_batch, entity_total, all_head_dicts=None, all_tail_dicts=None):
    negTripleList = [corrupt_head_filter(triple, entity_total, headDicts=all_head_dicts) if random.random() < 0.5 
        else corrupt_tail_filter(triple, entity_total, tailDicts=all_tail_dicts) for triple in triple_batch]
    # yield u, pi, ni, each list contains batch size ids,
    ph, pt, pr = getTripleElements(triple_batch)
    nh, nt, nr = getTripleElements(negTripleList)
    return ph, pt, pr, nh, nt, nr

# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter(triple, entityTotal, headDicts=None):
    while True:
        newHead = random.randrange(entityTotal)
        if newHead == triple[0] : continue
        if headDicts is not None:
            has_exist = False
            tr = (triple[1], triple[2])
            for head_dict in headDicts:
                if tr in head_dict and newHead in head_dict[tr]:
                    has_exist = True
                    break
            if has_exist: continue
            else: break
        else: break
    return (newHead, triple[1], triple[2])

# Change the tail of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_tail_filter(triple, entityTotal, tailDicts=None):
    while True:
        newTail = random.randrange(entityTotal)
        if newTail == triple[1] : continue
        if tailDicts is not None:
            has_exist = False
            hr = (triple[0], triple[2])
            for tail_dict in tailDicts:
                if hr in tail_dict and newTail in tail_dict[hr]:
                    has_exist = True
                    break
            if has_exist: continue
            else: break
        else: break
    return (triple[0], newTail, triple[2])

def getTripleElements(tripleList):
    headList = [triple[0] for triple in tripleList]
    tailList = [triple[1] for triple in tripleList]
    relList = [triple[2] for triple in tripleList]
    return headList, tailList, relList

def getNegRatings(ratingList, itemTotal, all_dicts=None):
    ni = []
    neg_set = set()
    for rating in ratingList:
        c_u = rating[0]
        oldItem = rating[1]
        # rating exists
        fliter_items = None
        if all_dicts is not None:
            fliter_items = set()
            for dic in all_dicts:
                if c_u in dic:
                    fliter_items.update(dic[c_u])
        while True:
            newItem = random.randrange(itemTotal)
            if newItem != oldItem and newItem not in fliter_items and newItem not in neg_set :
                break
        ni.append(newItem)
        neg_set.add(newItem)
    u = [rating[0] for rating in ratingList]
    pi = [rating[1] for rating in ratingList]
    return u, pi, ni

def MakeTrainIterator(
        train_data,
        batch_size,
        negtive_samples=1):
    train_list = np.array(train_data)

    def data_iter():
        dataset_size = len(train_list)
        order = list(range(dataset_size)) * negtive_samples
        random.shuffle(order)
        start = -1 * batch_size

        while True:
            start += batch_size
            if start > dataset_size - batch_size:
                # Start another epoch.
                start = 0
                random.shuffle(order)
            batch_indices = order[start:start + batch_size]
            
            # numpy
            yield train_list[batch_indices].tolist()
        
    return data_iter()

def MakeEvalIterator(
        eval_data,
        data_type,
        batch_size):
    # Make a list of minibatches from a dataset to use as an iterator.
    
    eval_list = np.asarray(eval_data, data_type)
    dataset_size = len(eval_list)
    order = list(range(dataset_size))
    data_iter = []
    start = -batch_size
    while True:
        start += batch_size

        if start >= dataset_size:
            break

        batch_indices = order[start:start + batch_size]
        candidate_batch = eval_list[batch_indices]
        data_iter.append(candidate_batch.tolist())

    return data_iter