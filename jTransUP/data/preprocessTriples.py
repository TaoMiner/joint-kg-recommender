import numpy as np
import json
import os
import random
import math
import logging

class Triple(object):
	def __init__(self, head, tail, relation):
		self.h = head
		self.t = tail
		self.r = relation

def splitRelationType(allTripleList):
    allHeadDict = {}
    allTailDict = {}
    for triple in allTripleList:
        tmp_head_set = allHeadDict.get( (triple.t, triple.r), set())
        tmp_head_set.add(triple.h)
        allHeadDict[(triple.t, triple.r)] = tmp_head_set

        tmp_tail_set = allTailDict.get( (triple.h, triple.r), set() )
        tmp_tail_set.add(triple.t)
        allTailDict[(triple.h, triple.r)] = tmp_tail_set
        
    one2oneRelations = set()
    one2manyRelations = set()
    many2oneRelations = set()
    many2manyRelations = set()
    
    rel_head_count_dict = {}
    rel_tail_count_dict = {}
    for er in allHeadDict:
        tmp_rel_count_list = rel_head_count_dict.get(er[1], [])
        tmp_rel_count_list.append(len(allHeadDict[er]))
        rel_head_count_dict[er[1]] = tmp_rel_count_list
    
    for er in allTailDict:
        tmp_rel_count_list = rel_tail_count_dict.get(er[1], [])
        tmp_rel_count_list.append(len(allTailDict[er]))
        rel_tail_count_dict[er[1]] = tmp_rel_count_list
    
    for r in rel_head_count_dict:
        avg_head_num = round( float( sum(rel_head_count_dict[r]) ) / len(rel_head_count_dict[r]) )
        avg_tail_num = round( float( sum(rel_tail_count_dict[r]) ) / len(rel_tail_count_dict[r]) )
        if avg_head_num > 1 and avg_tail_num > 1 :
            many2manyRelations.add(r)
        elif avg_head_num > 1 and avg_tail_num == 1:
            many2oneRelations.add(r)
        elif avg_head_num == 1 and avg_tail_num > 1:
            one2manyRelations.add(r)
        elif avg_head_num == 1 and avg_tail_num == 1:
            one2oneRelations.add(r)
        else:
            raise NotImplementedError
    return one2oneRelations, one2manyRelations, many2oneRelations, many2manyRelations

def splitKGData(triple_list, train_ratio = 0.7, test_ratio = 0.1, shuffle_data_split=False, filter_unseen_samples=True):
    # valid ratio could be 1-train_ratio-test_ratio, and maybe zero
    assert train_ratio > 0 and train_ratio < 1, "train ratio out of range!"
    assert test_ratio > 0 and test_ratio < 1, "test ratio out of range!"

    valid_ratio = 1 - train_ratio - test_ratio
    assert valid_ratio >= 0 and valid_ratio < 1, "valid ratio out of range!"

    train_ent_set = set()
    train_rel_set = set()

    if shuffle_data_split : random.shuffle(triple_list)

    n_total = len(triple_list)
    n_train = math.ceil(n_total * train_ratio)
    n_valid = math.ceil(n_total * valid_ratio) if valid_ratio > 0 else 0

    # in case of zero test item
    if n_train >= n_total:
        n_train = n_total - 1
        n_valid = 0
    elif n_train + n_valid >= n_total :
        n_valid = n_total - 1 - n_train

    tmp_train_list = [i for i in triple_list[0:n_train]]
    tmp_valid_list = [i for i in triple_list[n_train:n_train+n_valid]]
    tmp_test_list = [i for i in triple_list[n_train+n_valid:]]
    
    for triple in tmp_train_list:
        train_ent_set.add(triple[0])
        train_ent_set.add(triple[1])
        train_rel_set.add(triple[2])
    
    e_map = {}
    for index, ent in enumerate(train_ent_set):
        e_map[ent] = index
    r_map = {}
    for index, rel in enumerate(train_rel_set):
        r_map[rel] = index
    
    train_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_train_list]

    if filter_unseen_samples:
        valid_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_valid_list if triple[0] in train_ent_set and triple[1] in train_ent_set and triple[2] in train_rel_set]

        test_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_test_list if triple[0] in train_ent_set and triple[1] in train_ent_set and triple[2] in train_rel_set]
    else:
        valid_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_valid_list ]

        test_list = [Triple(e_map[triple[0]], e_map[triple[1]], r_map[triple[2]]) for triple in tmp_test_list ]
    # valid list length may be zero
    return train_list, valid_list, test_list, e_map, r_map

def parseRT(json_dict, ent_set=None, rel_set=None):
    r = json_dict['p']['value']
    t_type = json_dict['o']['type']
    t = json_dict['o']['value']
    if t_type != 'uri' or \
     ( ent_set is not None and t not in ent_set ) or \
     ( rel_set is not None and r not in rel_set ) :
        return None
    return r, t

def parseHR(json_dict, ent_set=None, rel_set=None):
    r = json_dict['p']['value']
    h = json_dict['s']['value']
    if  (ent_set is not None and h not in ent_set) or \
     ( rel_set is not None and r not in rel_set ) :
        return None
    return h, r

def loadRawData(filename, ent_vocab=None, rel_vocab=None, triple_list=[], ent_dic={}, logger=None):
    if logger is not None:
        ent_str = "use {} entities in vocab".format(len(ent_vocab)) if ent_vocab is not None else "no entity vocab provided"
        rel_str = "use {} relations in vocab".format(len(rel_vocab)) if rel_vocab is not None else "no relation vocab provided"
        logger.info("Predifined vocab: {}, and {}!".format(ent_str, rel_str))

    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) < 3 or \
            ( ent_vocab is not None and line_split[0] not in ent_vocab ) :
                continue
            e = line_split[0]

            count = ent_dic.get(e, 0)
            ent_dic[e] = count + 1

            head_json_list = json.loads(line_split[1])
            tail_json_list = json.loads(line_split[2])
            for head_json in head_json_list:
                rt = parseRT(head_json, ent_set=ent_vocab, rel_set=rel_vocab)
                if rt is None: continue
                r, t = rt
                count = ent_dic.get(t, 0)
                ent_dic[t] = count + 1
                triple_list.append( (e, t, r) )

            for tail_json in tail_json_list:
                hr = parseHR(tail_json, ent_set=ent_vocab, rel_set=rel_vocab)
                if hr is None: continue
                h, r = hr
                count = ent_dic.get(h, 0)
                ent_dic[h] = count + 1
                triple_list.append( (h, e, r) )

    if logger is not None:
        logger.info("Totally {} facts of {} entities from {}!".format(len(triple_list), len(ent_dic), filename))

    return triple_list, ent_dic

'''
def loadRelationTypes(filename):
    one2oneRelations = set()
    one2manyRelations = set()
    many2oneRelations = set()
    many2manyRelations = set()
    with open(filename, 'r') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) < 2 : continue
            if line_split[0] == 'one2one':
                tmp_set = one2oneRelations
            elif line_split[0] == 'one2many':
                tmp_set = one2manyRelations
            elif line_split[0] == 'many2one':
                tmp_set = many2oneRelations
            elif line_split[0] == 'many2many':
                tmp_set = many2manyRelations
            else:
                raise NotImplementedError
            for r_str in line_split[1:]:
                r = int(r_str)
                tmp_set.add(r)

    return one2oneRelations, one2manyRelations, many2oneRelations, many2manyRelations
'''

def cutLowFrequentData(triple_list, entity_frequency_dict, ent_vocab_to_keep=None, low_frequence=10):
    tmp_entity_set = set()
    tmp_relation_set = set()
    filtered_triple_list = []
    for triple in triple_list:
        if (entity_frequency_dict.get(triple[0], 0) >=low_frequence and entity_frequency_dict.get(triple[1], 0)>=low_frequence) or (triple[0] in ent_vocab_to_keep and triple[1] in ent_vocab_to_keep) or (triple[0] in ent_vocab_to_keep and entity_frequency_dict.get(triple[1], 0)>=low_frequence) or (entity_frequency_dict.get(triple[0], 0) >=low_frequence and triple[1] in ent_vocab_to_keep):
            filtered_triple_list.append(triple)
            tmp_entity_set.add(triple[0])
            tmp_entity_set.add(triple[1])
            tmp_relation_set.add(triple[2])
    return filtered_triple_list, tmp_entity_set, tmp_relation_set

def preprocess(triple_files, out_path, entity_file=None, relation_file=None, train_ratio=0.7, test_ratio=0.2, shuffle_data_split=True, filter_unseen_samples=True, low_frequence=10, logger=None):

    train_file = os.path.join(out_path, "train.dat")
    test_file = os.path.join(out_path, "test.dat")
    valid_file = os.path.join(out_path, "valid.dat") if 1 - train_ratio - test_ratio != 0 else None

    e_map_file = os.path.join(out_path, "e_map.dat")
    r_map_file = os.path.join(out_path, "r_map.dat")

    relation_type_file = os.path.join(out_path, "relation_type.dat")

    str_is_shuffle = "shuffle and split" if shuffle_data_split else "split without shuffle"
    if logger is not None:
        file_str = " ".join(triple_files)
        logger.info("{} {} for {:.1f} training, {:.1f} validation and {:.1f} testing!".format( str_is_shuffle, file_str, train_ratio, 1-train_ratio-test_ratio, test_ratio ))
    
    # predifined vocab for filtering
    ent_keep_vocab = None
    rel_vocab = None
    if relation_file is not None and os.path.exists(relation_file):
        rel_vocab = set()
        with open(relation_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                rel_vocab.add(line.strip())
    
    if entity_file is not None and os.path.exists(entity_file):
        ent_keep_vocab = set()
        with open(entity_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_split = line.strip().split('\t')
                if len(line_split) < 3 : continue
                ent_keep_vocab.add(line_split[2])

    triple_list = []
    ent_dic = {}
    for filename in triple_files:
        triple_list, ent_dic = loadRawData(filename, ent_vocab=None, rel_vocab=rel_vocab, triple_list=triple_list, ent_dic=ent_dic, logger=logger)

    # filter low frequent entities
    filtered_triple_list, e_set, r_set = cutLowFrequentData(triple_list, ent_dic, ent_vocab_to_keep=ent_keep_vocab, low_frequence=low_frequence)
    if logger is not None:
        logger.info("Cut infrequent entities (<={}), remaining {} facts of {} entities and {} relations!".format(low_frequence, len(filtered_triple_list), len(e_set), len(r_set) ) )

    train_list, valid_list, test_list, e_map, r_map = splitKGData(filtered_triple_list, train_ratio=train_ratio, test_ratio=test_ratio, shuffle_data_split=shuffle_data_split, filter_unseen_samples=filter_unseen_samples)

    one2oneRelations, one2manyRelations, many2oneRelations, many2manyRelations = splitRelationType(train_list + valid_list + test_list)

    if logger is not None:
        logger.debug("Spliting dataset and relation types are done!")
        logger.info("Filtering unseen entities and relations ..." if filter_unseen_samples else "Not filter unseen entities and relations.")
        logger.info("{} entities and {} relations, where {} train, {} valid, and {} test!".format(len(e_map), len(r_map), len(train_list), len(valid_list), len(test_list)))
        logger.info("where {} 1-1, {} 1-N, {} N-1, and {} N-N relations!".format(len(one2oneRelations), len(one2manyRelations), len(many2oneRelations), len(many2manyRelations) ))

    # save ent_dic, rel_dic
    with open(e_map_file, 'w', encoding='utf-8') as fout:
        for uri in e_map:
            fout.write('{}\t{}\n'.format(e_map[uri], uri))
    with open(r_map_file, 'w', encoding='utf-8') as fout:
        for uri in r_map:
            fout.write('{}\t{}\n'.format(r_map[uri], uri))
    with open(train_file, 'w', encoding='utf-8') as fout:
        for triple in train_list:
            fout.write('{}\t{}\t{}\n'.format(triple.h, triple.t, triple.r))
    with open(test_file, 'w', encoding='utf-8') as fout:
        for triple in test_list:
            fout.write('{}\t{}\t{}\n'.format(triple.h, triple.t, triple.r))
    
    if len(valid_list) > 0:
        with open(valid_file, 'w', encoding='utf-8') as fout:
            for triple in valid_list:
                fout.write('{}\t{}\t{}\n'.format(triple.h, triple.t, triple.r))
    
    with open(relation_type_file, 'w', encoding='utf-8') as fout:
        fout.write('one2one\t{}\n'.format('\t'.join([str(r) for r in one2oneRelations])))
        fout.write('one2many\t{}\n'.format('\t'.join([str(r) for r in one2manyRelations])))
        fout.write('many2one\t{}\n'.format('\t'.join([str(r) for r in many2oneRelations])))
        fout.write('many2many\t{}\n'.format('\t'.join([str(r) for r in many2manyRelations])))

def loadRelationType(type_file):
    with open(type_file, 'r', encoding='utf-8') as fin:
        typed_relations = [set(), set(), set(), set()]
        line_count = 0
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) <= 1 : continue
            for r in line_split[1:]:
                typed_relations[line_count].add(int(r))
            line_count += 1
    return typed_relations

def spliteTriples(test_file, relations, output_file):
    with open(test_file, 'r', encoding='utf-8') as fin:
        with open(output_file, 'w', encoding='utf-8') as fout:
            count = 0
            for line in fin:
                line_split = line.strip().split('\t')
                if len(line_split) != 3 : continue
                h = int(line_split[0])
                t = int(line_split[1])
                r = int(line_split[2])
                if r in relations:
                    fout.write("{}\t{}\t{}\n".format(h,t,r))
                    count += 1
    return count

if __name__ == "__main__":
    root_path = '/Users/caoyixin/Github/joint-kg-recommender/datasets/'
    dataset = 'dbbook2014'

    relation_type_file = root_path + dataset +'/kg/relation_type.dat'

    test_file = root_path + dataset +'/kg/test.dat'
    log_file = root_path + dataset + '/data_preprocess.log'
    one2one_file = root_path + dataset +'/kg/one2one.dat'
    one2N_file = root_path + dataset +'/kg/one2N.dat'
    N2one_file = root_path + dataset +'/kg/N2one.dat'
    N2N_file = root_path + dataset +'/kg/N2N.dat'

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

    typed_relations = loadRelationType(relation_type_file)

    one2one_count = spliteTriples(test_file, typed_relations[0], one2one_file)
    logger.info("generate {} 1-to-1 triples of {} relations!".format(one2one_count, len(typed_relations[0])))

    one2N_count = spliteTriples(test_file, typed_relations[1], one2N_file)
    logger.info("generate {} 1-to-N triples of {} relations!".format(one2N_count, len(typed_relations[1])))

    N2one_count = spliteTriples(test_file, typed_relations[2], N2one_file)
    logger.info("generate {} N-to-1 triples of {} relations!".format(N2one_count, len(typed_relations[2])))

    N2N_count = spliteTriples(test_file, typed_relations[3], N2N_file)
    logger.info("generate {} N-to-N triples of {} relations!".format(N2N_count, len(typed_relations[3])))