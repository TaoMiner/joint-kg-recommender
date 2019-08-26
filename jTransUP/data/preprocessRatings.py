import numpy as np
import pandas as pd
import csv
import json
import os
import random
import math
import logging

class Rating(object):
	def __init__(self, user, item, rating):
		self.u = user
		self.i = item
		self.r = rating

def splitRatingData(user_dict, train_ratio = 0.7, test_ratio = 0.2, shuffle_data_split=False, filter_unseen_samples=True):
    # valid ratio could be 1-train_ratio-test_ratio, and maybe zero
    
    assert train_ratio > 0 and train_ratio < 1, "train ratio out of range!"
    assert test_ratio > 0 and test_ratio < 1, "test ratio out of range!"

    valid_ratio = 1 - train_ratio - test_ratio
    assert valid_ratio >= 0 and valid_ratio < 1, "valid ratio out of range!"

    train_item_set = set()
    tmp_train_list = []
    tmp_valid_list = []
    tmp_test_list = []
    for user in user_dict:
        tmp_item_list = user_dict[user]

        n_items = len(tmp_item_list)
        n_train = math.ceil(n_items * train_ratio)
        n_valid = math.ceil(n_items * valid_ratio) if valid_ratio > 0 else 0
        # in case of zero test item
        if n_train >= n_items:
            n_train = n_items - 1
            n_valid = 0
        elif n_train + n_valid >= n_items :
            n_valid = n_items - 1 - n_train

        if shuffle_data_split : random.shuffle(tmp_item_list)
        
        for ir in tmp_item_list[0:n_train]:
            tmp_train_list.append( (user, ir[0], ir[1]) )
            train_item_set.add(ir[0])
        tmp_valid_list.extend([(user, ir[0], ir[1]) for ir in tmp_item_list[n_train:n_train+n_valid]])

        tmp_test_list.extend( [(user, ir[0], ir[1]) for ir in tmp_item_list[n_train+n_valid:]] )

    u_map = {}
    for index, user in enumerate(user_dict.keys()):
        u_map[user] = index
    i_map = {}
    for index, item in enumerate(train_item_set):
        i_map[item] = index

    train_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_train_list]
    
    if filter_unseen_samples:
        valid_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_valid_list if rating[1] in train_item_set]

        test_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_test_list if rating[1] in train_item_set]
    else:
        valid_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_valid_list ]

        test_list = [Rating(u_map[rating[0]], i_map[rating[1]], rating[2]) for rating in tmp_test_list ]

    return train_list, valid_list, test_list, u_map, i_map

def cutLowFrequentData(rating_file, item_vocab=None, low_frequence=10, logger=None):
    df = pd.read_csv(rating_file, encoding='utf-8')
    df = df[['userId', 'itemId', 'rating']]
    df = df.values

    user_dict = dict()
    item_dict = dict()

    f_user_dict = dict()
    f_item_dict = dict()

    for line in df:
        u_id = int(line[0])
        i_id = int(line[1])
        r_score = int(line[2])

        if item_vocab is not None and i_id not in item_vocab : continue

        if u_id in user_dict:
            user_dict[u_id].append( (i_id, r_score) )
        else:
            user_dict[u_id] = [(i_id, r_score)]

        if i_id in item_dict.keys():
            item_dict[i_id].append( (u_id, r_score) )
        else:
            item_dict[i_id] = [(u_id, r_score)]

    if logger is not None:
        logger.info("Totally {} interactions between {} user and {} items!".format(len(df), len(user_dict), len(item_dict)))
        logger.debug("Filtering infrequent users and items (<={}) ...".format(low_frequence))
    while True:
        flag1, flag2 = True, True
        
        for u_id in user_dict.keys():
            pos_items = user_dict[u_id]
            valid_items = [idx for idx in pos_items if idx[0] in item_dict.keys()]

            if len(valid_items) >= low_frequence:
                f_user_dict[u_id] = valid_items
            else:
                flag1 = False
        
        total_ratings = 0
        for i_id in item_dict.keys():
            pos_users = item_dict[i_id]
            valid_users = [udx for udx in pos_users if udx[0] in user_dict.keys()]

            if len(valid_users) >= low_frequence:
                f_item_dict[i_id] = valid_users
                total_ratings += len(valid_users)
            else:
                flag2 = False

        user_dict = f_user_dict.copy()
        item_dict = f_item_dict.copy()
        f_user_dict = {}
        f_item_dict = {}

        if logger is not None:
            logger.info("Remaining : {} interactions of {} users and {} items!".format( total_ratings, len(user_dict), len(item_dict)))
        if flag1 and flag2:
            if logger is not None: logger.debug('Filtering infrequent users and items done!')
            break

    return user_dict

def preprocess(rating_file, out_path, train_ratio=0.7, test_ratio=0.2, shuffle_data_split=True, filter_unseen_samples=True, low_frequence=10, logger=None):
    train_file = os.path.join(out_path, "train.dat")
    test_file = os.path.join(out_path, "test.dat")
    valid_file = os.path.join(out_path, "valid.dat") if 1 - train_ratio - test_ratio != 0 else None

    u_map_file = os.path.join(out_path, "u_map.dat")
    i_map_file = os.path.join(out_path, "i_map.dat")

    str_is_shuffle = "shuffle and split" if shuffle_data_split else "split without shuffle"

    if logger is not None:
        logger.info("{} {} for {:.1f} training, {:.1f} validation and {:.1f} testing!".format( str_is_shuffle, rating_file, train_ratio, 1-train_ratio-test_ratio, test_ratio ))
    
    # only remain the items in the item_vocab
    item_vocab = None

    user_dict = cutLowFrequentData(rating_file, item_vocab=item_vocab, low_frequence=low_frequence, logger=logger)

    train_list, valid_list, test_list, u_map, i_map = splitRatingData(user_dict, train_ratio = train_ratio, test_ratio = test_ratio, shuffle_data_split=shuffle_data_split, filter_unseen_samples=filter_unseen_samples)

    if logger is not None:
        logger.debug("Spliting dataset is done!")
        logger.info("Filtering unseen users and items ..." if filter_unseen_samples else "Not filter unseen users and items.")
        logger.info("{} users and {} items, where {} train, {} valid, and {} test!".format(len(u_map), len(i_map), len(train_list), len(valid_list), len(test_list)))
        
    # save ent_dic, rel_dic 
    with open(u_map_file, 'w', encoding='utf-8') as fout:
        for org_u_id in u_map:
            fout.write('{}\t{}\n'.format(u_map[org_u_id], org_u_id))

    with open(i_map_file, 'w', encoding='utf-8') as fout:
        for org_i_id in i_map:
            fout.write('{}\t{}\n'.format(i_map[org_i_id], org_i_id))

    with open(train_file, 'w', encoding='utf-8') as fout:
        for rating in train_list:
            fout.write('{}\t{}\t{}\n'.format(rating.u, rating.i, rating.r))

    with open(test_file, 'w', encoding='utf-8') as fout:
        for rating in test_list:
            fout.write('{}\t{}\t{}\n'.format(rating.u, rating.i, rating.r))
    
    if len(valid_list) > 0:
        with open(valid_file, 'w', encoding='utf-8') as fout:
            for rating in valid_list:
                fout.write('{}\t{}\t{}\n'.format(rating.u, rating.i, rating.r))

def loadRatings(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        user_dict = {}
        total_count = 0
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3 : continue
            u = int(line_split[0])
            i = int(line_split[1])
            rating = int(line_split[2])

            i_set = user_dict.get(u, set())
            i_set.add( (i, rating) )
            user_dict[u] = i_set
            total_count += 1
    return total_count, user_dict

def getMaxMinRatings(user_dict):
    max_ratings = 0
    min_ratings = 10000
    for u in user_dict:
        if len(user_dict[u]) > max_ratings:
            max_ratings = len(user_dict[u])
        if len(user_dict[u]) < min_ratings:
            min_ratings = len(user_dict[u])
    return max_ratings, min_ratings

def splitUsers(user_dict, split_num):
    split_num = 10
    splited_users = [set() for _ in range(split_num)]
    max_ratings, min_ratings = getMaxMinRatings(user_dict)
    step = math.ceil((max_ratings - min_ratings + 1) / split_num)
    splited_threshold = [i for i in range(min_ratings, max_ratings, step) if i!= min_ratings] + [max_ratings]
    splited_threshold = [20, 40, 60, 80, 100, 150, 200, 300, 400, max_ratings]
    
    for u in user_dict:
        rating_num = len(user_dict[u])
        for i, thr in enumerate(splited_threshold):
            if rating_num <= thr:
                splited_users[i].add(u)
                break
    return splited_threshold, splited_users

def output(filename, user_dict, u_ids):
    count = 0
    with open(filename, 'w', encoding='utf-8') as fout:
        for u in user_dict:
            if u in u_ids:
                for ir in user_dict[u]:
                    fout.write("{}\t{}\t{}\n".format(u, ir[0], ir[1]))
                    count += 1
    return count

if __name__ == "__main__":
    root_path = '/Users/caoyixin/Github/joint-kg-recommender/datasets/'
    dataset = 'ml1m'
    split_num = 10
    train_file = root_path + dataset +'/train.dat'
    test_file = root_path + dataset +'/test.dat'
    log_file = root_path + dataset + '/data_preprocess.log'

    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    train_rating_total, train_user_dict = loadRatings(train_file)
    test_rating_total, test_user_dict = loadRatings(test_file)

    max_ratings, min_ratings = getMaxMinRatings(train_user_dict)
    logger.info("load {} ratings for {} users, where min {} and max {} ratings!".format(train_rating_total, len(train_user_dict), min_ratings, max_ratings))
    
    splited_threshold, splited_users = splitUsers(train_user_dict, split_num)

    for i, u_ids in enumerate(splited_users):
        logger.info("generating test ratings if {} user ratings num < {} ...".format(len(u_ids), splited_threshold[i]))
        filename = root_path + dataset +'/test{}.dat'.format(i)
        count = output(filename, test_user_dict, u_ids)
        logger.info("output {} ratings done!".format(count))