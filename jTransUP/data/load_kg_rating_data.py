import os
from jTransUP.data import load_rating_data, load_triple_data

# two items refer to the same entity
def loadR2KgMap(filename):
    i2kg_map = {}
    kg2i_map = {}
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3 : continue
            i_id = line_split[0]
            kg_uri = line_split[2]
            i2kg_map[i_id] = kg_uri
            kg2i_map[kg_uri] = i_id
    print("successful load {} item and {} entity pairs!".format(len(i2kg_map), len(kg2i_map)))
    return i2kg_map, kg2i_map

# map: org:id
# link: org(map1):org(map2)
def rebuildEntityItemVocab(map1, map2, links):
    new_map = {}
    index = 0
    has_map2 = {}
    remap1 = {}
    for org_id1 in map1:
        mapped_id2 = -1
        if org_id1 in links:
            org_id2 = links[org_id1]
            if org_id2 in map2:
                mapped_id2 = map2[org_id2]
                has_map2[org_id2] = index
        new_map[index] = (map1[org_id1], mapped_id2)
        
        remap1[map1[org_id1]] = index
        index += 1

    remap2 = {}
    mapped_id1 = -1
    for org_id2 in map2:
        if org_id2 in has_map2 :
            remap2[map2[org_id2]] = has_map2[org_id2]
            continue
        new_map[index] = (mapped_id1, map2[org_id2])

        remap2[map2[org_id2]] = index
        index += 1
    return new_map, remap1, remap2, len(has_map2)
            

def load_data(data_path, rec_eval_files, kg_eval_files, batch_size, negtive_samples=1, logger=None):
    kg_path = os.path.join(data_path, 'kg')
    map_file = os.path.join(data_path, 'i2kg_map.tsv')

    rating_train_dataset, rating_eval_datasets, u_map, i_map = load_rating_data.load_data(data_path, rec_eval_files, batch_size, logger=logger, negtive_samples=negtive_samples)

    triple_train_dataset, triple_eval_datasets, e_map, r_map = load_triple_data.load_data(kg_path, kg_eval_files, batch_size, logger=logger, negtive_samples=negtive_samples)

    i2kg_map, kg2i_map = loadR2KgMap(map_file)
    # e_map,imap org--> new id
    ikg_map, e_remap, i_remap, aligned_ie_total = rebuildEntityItemVocab(e_map, i_map, kg2i_map)

    if logger is not None:
        logger.info("Find {} aligned items and entities!".format(aligned_ie_total))

    return rating_train_dataset, rating_eval_datasets, u_map, i_remap, triple_train_dataset, triple_eval_datasets, e_remap, r_map, ikg_map