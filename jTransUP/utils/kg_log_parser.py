import re

# two items refer to the same entity
def loadR2KgMap(filename, item_vocab=None, kg_vocab=None):
    i2kg_map = {}
    kg2i_map = {}
    with open(filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3 : continue
            i_id = int(line_split[0])
            kg_uri = line_split[2]
            if item_vocab is not None and kg_vocab is not None:
                if i_id not in item_vocab or kg_uri not in kg_vocab : continue
                i_id = item_vocab[i_id]
                kg_uri = kg_vocab[kg_uri]
            i2kg_map[i_id] = kg_uri
            kg2i_map[kg_uri] = i_id
    print("successful load {} item and {} entity pairs!".format(len(i2kg_map), len(kg2i_map)))
    return i2kg_map, kg2i_map

def loadRecVocab(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        vocab = {}
        vocab_reverse = {}
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 2 : continue
            mapped_id = int(line_split[0])
            org_id = int(line_split[1])
            vocab[org_id] = mapped_id
            vocab_reverse[mapped_id] = org_id
        print("load {} vocab!".format(len(vocab)))

    return vocab, vocab_reverse

def loadKGVocab(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        vocab = {}
        vocab_reverse = {}
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 2 : continue
            mapped_id = int(line_split[0])
            org_id = line_split[1]
            vocab[org_id] = mapped_id
            vocab_reverse[mapped_id] = org_id
        print("load {} vocab!".format(len(vocab)))
    return vocab, vocab_reverse

def loadTriples(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        triple_total = 0
        triple_list = []
        triple_head_dict = {}
        triple_tail_dict = {}
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3 : continue
            h_id = int(line_split[0])
            t_id = int(line_split[1])
            r_id = int(line_split[2])

            triple_list.append( (h_id, t_id, r_id) )

            tmp_heads = triple_head_dict.get( (t_id, r_id), set())
            tmp_heads.add(h_id)
            triple_head_dict[(t_id, r_id)] = tmp_heads

            tmp_tails = triple_tail_dict.get( (h_id, r_id), set())
            tmp_tails.add(t_id)
            triple_tail_dict[(h_id, r_id)] = tmp_tails

            triple_total += 1

    return triple_total, triple_list, triple_head_dict, triple_tail_dict

def parseKGResults(log_filename):
    head_results = {}
    tail_results = {}
    head_correct_num = 0
    tail_correct_num = 0
    head_wrong_num = 0
    tail_wrong_num = 0
    with open(log_filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 5: continue
            is_head = line_split[0].endswith('H')
            h_id = int(line_split[1])
            t_id = int(line_split[2])
            r_id = int(line_split[3])
            is_correct = line_split[4]=='1'

            if is_head:
                tmp_results = head_results.get((t_id,r_id), [set(), set()])
                i = 0 if is_correct else 1
                tmp_results[i].add(h_id)
                head_results[(t_id,r_id)] = tmp_results
                head_wrong_num += i
                head_correct_num += (1-i)
            else:
                tmp_results = tail_results.get((h_id,r_id), [set(), set()])
                i = 0 if is_correct else 1
                tmp_results[i].add(t_id)
                tail_results[(h_id,r_id)] = tmp_results
                tail_wrong_num += i
                tail_correct_num += (1-i)
            
    print("parse {} head predictions, avgerage {} correct and {} wrong!".format(len(head_results), head_correct_num/len(head_results), head_wrong_num/len(head_results)))
    print("parse {} tail predictions, avgerage {} correct and {} wrong!".format(len(tail_results), tail_correct_num/len(tail_results), tail_wrong_num/len(tail_results)))

    return head_results, tail_results

rel_p = re.compile(r'([\d]+)\(([\d]+)\)')
def parseRecResults(log_filename):
    rating_dict = {}
    rel_set = set()
    total_num = 0
    with open(log_filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3: continue
            u_id = int(line_split[0].split('user:')[-1])

            gold_ids = [s for s in re.split(r':|,', line_split[1])[1:]]
            for i_id in gold_ids:
                m = rel_p.match(i_id)
                if m is not None:
                    gold_id = int(m[1])
                    rel_id = int(m[2])
                    rel_set.add(rel_id)
                    tmp_u_ids = rating_dict.get((gold_id,rel_id), set())
                    tmp_u_ids.add(u_id)
                    rating_dict[(gold_id,rel_id)] = tmp_u_ids
                    total_num += 1
    print("parse {} item/rel pairs, avgerage {} users, invovled {} relations!".format(len(rating_dict), total_num/len(rating_dict), len(rel_set)))

    return rating_dict, rel_set

# rating dict to search (item,r) --> user
def compareLogs(log1, log2, kg2i_map, i2kg_map, rating_dict, triple_head_dict, triple_tail_dict, output_file, rel_set=None, u_map_reverse=None, i_map_reverse=None, e_map_reverse=None, r_map_reverse=None):

    head_results1, tail_results1 = parseKGResults(log1)
    head_results2, tail_results2 = parseKGResults(log2)

    with open(output_file, 'w', encoding='utf-8') as fout:
        for tr in head_results1:
            if tr not in head_results2 : continue
            correct1, wrong1 = head_results1[tr]
            correct2, wrong2 = head_results2[tr]

            target_heads = correct2 - correct1
            if len(target_heads) == 0 or tr not in triple_head_dict: continue

            gold_heads = triple_head_dict[tr]

            outputHead(fout, tr, target_heads, kg2i_map, i2kg_map, gold_heads, rating_dict, rel_set=rel_set, u_map_reverse=u_map_reverse, i_map_reverse=i_map_reverse, e_map_reverse=e_map_reverse, r_map_reverse=r_map_reverse)

        for hr in tail_results1:
            if hr not in tail_results2 : continue
            correct1, wrong1 = tail_results1[hr]
            correct2, wrong2 = tail_results2[hr]

            target_tails = correct2 - correct1
            if len(target_tails) == 0 or hr not in triple_tail_dict: continue

            gold_tails = triple_tail_dict[hr]
            
            outputTail(fout, hr, target_tails, kg2i_map, i2kg_map, gold_tails, rating_dict, rel_set=rel_set, u_map_reverse=u_map_reverse, i_map_reverse=i_map_reverse, e_map_reverse=e_map_reverse, r_map_reverse=r_map_reverse)

# rating dict: {}
def outputHead(fout, tr, target_heads, kg2i_map, i2kg_map, gold_heads, rating_dict, rel_set=None, u_map_reverse=None, i_map_reverse=None, e_map_reverse=None, r_map_reverse=None):
    out_reverse = False
    if u_map_reverse is not None and i_map_reverse is not None and e_map_reverse is not None and r_map_reverse is not None :
        out_reverse = True
    t = tr[0]
    r = tr[1]
    if out_reverse:
        out_str = "tail:{}\trel:{}\ttarget_head:[{}]".format(e_map_reverse[t], r_map_reverse[r], ",".join([e_map_reverse[h] for h in target_heads]))
    else:
        out_str = "tail:{}\trel:{}\ttarget_head:[{}]".format(e_map_reverse[t], r_map_reverse[r], ",".join([str(h) for h in target_heads]))

    remap_hs = [kg2i_map.get(h, -1) for h in target_heads]
    tmp_gold_hs = gold_heads - target_heads
    remap_ghs = [kg2i_map.get(gold_h, -1) for gold_h in tmp_gold_hs]

    for h in remap_hs:
        u_ids = rating_dict.get((h, r), set())
        if len(u_ids) == 0: continue
        for gh in remap_ghs:
            gu_ids = rating_dict.get((gh, r), set())
            if len(gu_ids) == 0: continue
            target_users = u_ids & gu_ids
            if len(target_users) > 0:
                if out_reverse:
                    tmp_str = "{},{},[{}]".format(e_map_reverse[i2kg_map[h]], e_map_reverse[i2kg_map[gh]], ",".join([str(u_map_reverse[u]) for u in target_users]))
                else:
                    tmp_str = "{},{},[{}]".format(h, gh, ",".join([str(u) for u in target_users]))
                out_str += "\tusers:{}".format(tmp_str)
        if rel_set is not None:
            tmp_rels = rel_set - r
            for tmp_r in tmp_rels:
                u_ids = rating_dict.get((h, tmp_r), set())
                if len(u_ids) == 0: continue
                for gh in remap_ghs:
                    gu_ids = rating_dict.get((gh, tmp_r), set())
                    if len(gu_ids) == 0: continue
                    target_users = u_ids & gu_ids
                    if len(target_users) > 0:
                        if out_reverse:
                            tmp_str = "{},{},{},[{}]".format(e_map_reverse[i2kg_map[h]], e_map_reverse[i2kg_map[gh]], r_map_reverse[tmp_r], ",".join([u_map_reverse[u] for u in target_users]))
                        else:
                            tmp_str = "{},{},{},[{}]".format(h, gh, r_map_reverse[tmp_r], ",".join([str(u) for u in target_users]))
                        out_str += "\tusers:{}".format(tmp_str)
    fout.write("{}\n".format(out_str))

def outputTail(fout, hr, target_tails, kg2i_map, i2kg_map, gold_tails, rating_dict, rel_set=None, u_map_reverse=None, i_map_reverse=None, e_map_reverse=None, r_map_reverse=None):
    out_reverse = False
    if u_map_reverse is not None and i_map_reverse is not None and e_map_reverse is not None and r_map_reverse is not None :
        out_reverse = True
    h = hr[0]
    r = hr[1]
    if out_reverse:
        out_str = "head:{}\trel:{}\ttarget_tail:{}".format(e_map_reverse[h], r_map_reverse[r], ",".join([e_map_reverse[t] for t in target_tails]))
    else:
        out_str = "head:{}\trel:{}\ttarget_tail:{}".format(e_map_reverse[h], r_map_reverse[r], ",".join([str(t) for t in target_tails]))

    remap_ts = [kg2i_map.get(t, -1) for t in target_tails]
    tmp_gold_ts = gold_tails - target_tails
    remap_gts = [kg2i_map.get(gold_t, -1) for gold_t in tmp_gold_ts]

    for t in remap_ts:
        u_ids = rating_dict.get((t, r), set())
        if len(u_ids) == 0: continue
        for gt in remap_gts:
            gu_ids = rating_dict.get((gt, r), set())
            if len(gu_ids) == 0: continue
            target_users = u_ids & gu_ids
            if len(target_users) > 0:
                if out_reverse:
                    tmp_str = "{},{},[{}]".format(e_map_reverse[i2kg_map[t]], e_map_reverse[i2kg_map[gt]], ",".join([str(u_map_reverse[u]) for u in target_users]))
                else:
                    tmp_str = "{},{},[{}]".format(t, gt, ",".join([str(u) for u in target_users]))
                out_str += "\tusers:{}".format(tmp_str)
        if rel_set is not None:
            tmp_rels = rel_set - r
            for tmp_r in tmp_rels:
                u_ids = rating_dict.get((t, tmp_r), set())
                if len(u_ids) == 0: continue
                for gt in remap_gts:
                    gu_ids = rating_dict.get((gt, tmp_r), set())
                    if len(gu_ids) == 0: continue
                    target_users = u_ids & gu_ids
                    if len(target_users) > 0:
                        if out_reverse:
                            tmp_str = "{},{},{},[{}]".format(e_map_reverse[i2kg_map[t]], e_map_reverse[i2kg_map[gt]], r_map_reverse[tmp_r], ",".join([str(u_map_reverse[u]) for u in target_users]))
                        else:
                            tmp_str = "{},{},{},[{}]".format(t, gt, r_map_reverse[tmp_r], ",".join([str(u) for u in target_users]))
                        out_str += "\tusers:{}".format(tmp_str)
    fout.write("{}\n".format(out_str))


root_path = '/Users/caoyixin/Github/joint-kg-recommender'
dataset_path = root_path + '/datasets/ml1m'

transh_log = root_path + '/log/ml1m-transh-eval.log'
jtransup_log = root_path + '/log/ml1m-jtransup_share-eval.log'

u_map_file = dataset_path + '/u_map.dat'
i_map_file = dataset_path + '/i_map.dat'
e_map_file = dataset_path + '/kg/e_map.dat'
r_map_file = dataset_path + '/kg/r_map.dat'
i2kg_map_file = dataset_path + '/i2kg_map.tsv'
train_triple_file = dataset_path + '/kg/train.dat'
output_file = root_path + '/log/parse_transh_jtransup.log'

user_vocab, user_vocab_reverse = loadRecVocab(u_map_file)
item_vocab, item_vocab_reverse = loadRecVocab(i_map_file)
kg_vocab, kg_vocab_reverse = loadKGVocab(e_map_file)
rel_vocab, rel_vocab_reverse = loadKGVocab(r_map_file)
i2kg_map, kg2i_map = loadR2KgMap(i2kg_map_file, item_vocab=item_vocab, kg_vocab=kg_vocab)

triple_total, triple_list, triple_head_dict, triple_tail_dict = loadTriples(train_triple_file)

rating_dict, rel_set = parseRecResults(jtransup_log)

compareLogs(transh_log, jtransup_log, kg2i_map, i2kg_map, rating_dict, triple_head_dict, triple_tail_dict, output_file, rel_set=None, u_map_reverse=user_vocab_reverse, i_map_reverse=item_vocab_reverse, e_map_reverse=kg_vocab_reverse, r_map_reverse=rel_vocab_reverse)

