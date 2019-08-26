import os
import numpy as np
from jTransUP.utils.data import MakeTrainIterator, MakeEvalIterator

# org--> id
def loadVocab(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        vocab = {}
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 2 : continue
            mapped_id = int(line_split[0])
            org_id = line_split[1]
            vocab[org_id] = mapped_id

    return vocab

# dict:{u_id:set(i_ids), ... }
def loadRatings(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        rating_total = 0
        rating_list = []
        rating_dict = {}
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3 : continue
            u_id = int(line_split[0])
            i_id = int(line_split[1])
            r_score = int(line_split[2])
            rating_list.append( (u_id, i_id) )

            tmp_items = rating_dict.get(u_id, set())
            tmp_items.add(i_id)
            rating_dict[u_id] = tmp_items

            rating_total += 1

    return rating_total, rating_list, rating_dict

def load_data(data_path, eval_filenames, batch_size, negtive_samples=1, logger=None):

    train_file = os.path.join(data_path, "train.dat")

    eval_files = []
    for file_name in eval_filenames:
        eval_files.append(os.path.join(data_path, file_name))

    u_map_file = os.path.join(data_path, "u_map.dat")
    i_map_file = os.path.join(data_path, "i_map.dat")

    train_total, train_list, train_dict = loadRatings(train_file)

    eval_dataset = []
    for eval_file in eval_files:
        eval_dataset.append( loadRatings(eval_file) )
    
    if logger is not None:
        eval_totals = [str(eval_data[0]) for eval_data in eval_dataset]
        logger.info("Totally {} train ratings, {} eval ratings in files: {}!".format(train_total, ",".join(eval_totals), ";".join(eval_files)))
    
    # get user total
    u_map = loadVocab(u_map_file)
    # get item total
    i_map = loadVocab(i_map_file)
    
    if logger is not None:
        logger.info("successfully load {} users and {} items!".format(len(u_map), len(i_map)))
    
    train_iter = MakeTrainIterator(train_list, batch_size, negtive_samples=negtive_samples)

    new_eval_datasets = []
    dt = np.dtype('int')
    for eval_data in eval_dataset:
        tmp_iter = MakeEvalIterator(list(eval_data[2].keys()), dt, batch_size)
        new_eval_datasets.append([tmp_iter, eval_data[0], eval_data[1], eval_data[2]])

    train_dataset = (train_iter, train_total, train_list, train_dict)
    return train_dataset, new_eval_datasets, u_map, i_map

if __name__ == "__main__":
    # Demo:
    data_path = "/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/"
    batch_size = 10
    from jTransUP.data.load_kg_rating_data import loadR2KgMap

    i2kg_file = os.path.join(data_path, 'i2kg_map.tsv')
    i2kg_pairs = loadR2KgMap(i2kg_file)
    i_set = set([p[0] for p in i2kg_pairs])

    datasets, rating_iters, u_map, i_map, user_total, item_total = load_data(data_path, batch_size, item_vocab=i_set)

    trainList, testDict, validDict, allDict, testTotal, validTotal = datasets
    print("user:{}, item:{}!".format(user_total, item_total))
    print("totally ratings for {} train, {} valid, and {} test!".format(len(trainList), item_total, testTotal))