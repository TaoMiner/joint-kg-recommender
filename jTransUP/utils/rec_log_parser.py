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

rel_p = re.compile(r'([\d]+)\(([\d]+)\)')
def parseRecResults(log_filename, model_type):
    results = {}
    user_item_relations = {}
    correct_num = 0
    wrong_num = 0
    with open(log_filename, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) != 3: continue
            u_id = int(line_split[0].split('user:')[-1])
            pred_ids = set([int(i) for i in re.split(r':|,', line_split[2])[1:]])

            if model_type not in ['transup', 'jtransup']:
                gold_ids = set([int(i) for i in re.split(r':|,', line_split[1])[1:]])
            else:
                tmp_gold_ids = [s for s in re.split(r':|,', line_split[1])[1:]]
                gold_ids = set()
                for s in tmp_gold_ids:
                    m = rel_p.match(s)
                    if m is not None:
                        gold_id = int(m[1])
                        rel_id = int(m[2])
                        gold_ids.add(gold_id)
                        user_item_relations[(u_id,gold_id)] = rel_id
            correct = pred_ids & gold_ids
            wrong = pred_ids - correct
            correct_num += len(correct)
            wrong_num += len(wrong)
            results[u_id] = (correct, wrong, gold_ids)
    print("parse {} users, avgerage {} correct and {} wrong!".format(len(results), correct_num/len(results), wrong_num/len(results)))
    print("parse {} user item relations!".format(len(user_item_relations)))

    return results, user_item_relations

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

def compareLogs(log1, log2, model1, model2, i2kg_map, triple_head_dict, triple_tail_dict, output_file, u_map_reverse=None, i_map_reverse=None, e_map_reverse=None, r_map_reverse=None, users=None):
    results1, preference1 = parseRecResults(log1, model1)
    results2, preference2 = parseRecResults(log2, model2)
    with open(output_file, 'w', encoding='utf-8') as fout:
        for u_id in results1:
            if u_id not in results2 or (users is not None and u_map_reverse[u_id] not in users): continue
            correct1, wrong1, gold_ids1 = results1[u_id]
            correct2, wrong2, gold_ids2 = results2[u_id]
            target_items = correct2 - correct1
            # if len(target_items) == 0 : continue
            analysis(fout, u_id, target_items, preference2, gold_ids2, i2kg_map, triple_head_dict, triple_tail_dict, u_map_reverse=u_map_reverse, i_map_reverse=i_map_reverse, e_map_reverse=e_map_reverse, r_map_reverse=r_map_reverse)

def analysis(fout, u_id, target_items, preference, gold_ids, i2kg_map, triple_head_dict, triple_tail_dict, u_map_reverse=None, i_map_reverse=None, e_map_reverse=None, r_map_reverse=None):
    out_reverse = False
    if u_map_reverse is not None and i_map_reverse is not None and e_map_reverse is not None and r_map_reverse is not None :
        out_reverse = True
    if out_reverse:
        out_str = "user:{}\ttarget:{}".format(u_map_reverse[u_id], ",".join([e_map_reverse[i2kg_map[ti]] for ti in target_items if ti in i2kg_map]))
    else:
        out_str = "user:{}\ttarget:{}".format(u_id, ",".join([str(ti) for ti in target_items]))

    target_rel_ids = [preference.get((u_id, ti), -1) for ti in target_items]
    tmp_gold_ids = gold_ids

    gold_rel_ids = [preference.get((u_id, gi), -1) for gi in tmp_gold_ids]
    u_prefs_target = set(target_rel_ids) - set([-1])
    u_prefs_gold = set(gold_rel_ids) - set([-1])

    share_prefs = u_prefs_target & u_prefs_gold
    if len(share_prefs) == 0: return None
    org_pref = [r_map_reverse[r] for r in share_prefs]
    if 'http://dbpedia.org/ontology/starring' not in org_pref or 'http://dbpedia.org/ontology/director' not in org_pref: return None
    remap_tis = [i2kg_map.get(ti, -1) for ti in target_items]
    remap_gids = [i2kg_map.get(gold_id, -1) for gold_id in tmp_gold_ids]
    for tr in zip(remap_tis, target_rel_ids):
        if tr[1] not in share_prefs : continue
        h_ids = triple_head_dict.get(tr, set())
        t_ids = triple_tail_dict.get(tr, set())
        if len(h_ids) == 0 and len(t_ids) == 0 : continue
        for gid in remap_gids:
            if gid == tr[0] : continue
            gh_ids = triple_head_dict.get((gid, tr[1]), set())
            gt_ids = triple_tail_dict.get((gid, tr[1]), set())
            if len(gh_ids) == 0 and len(gt_ids) == 0 : continue
            h_gh = h_ids & gh_ids
            h_gt = h_ids & gt_ids
            t_gh = t_ids & gh_ids
            t_gt = t_ids & gt_ids
            if len(h_gh) > 0:
                if out_reverse:
                    tmp_str = "{},{},{},[{}]".format(e_map_reverse[tr[0]], e_map_reverse[gid], r_map_reverse[tr[1]], ",".join([str(e_map_reverse[i]) for i in h_gh]))
                else:
                    tmp_str = "{},{},{},[{}]".format(tr[0], gid, tr[1], ",".join([str(i) for i in h_gh]))
                out_str += "\nhh:{}".format(tmp_str)
            if len(h_gt) > 0:
                if out_reverse:
                    tmp_str = "{},{},{},[{}]".format(e_map_reverse[tr[0]], e_map_reverse[gid], r_map_reverse[tr[1]], ",".join([str(e_map_reverse[i]) for i in h_gt]))
                else:
                    tmp_str = "{},{},{},[{}]".format(tr[0], gid, tr[1], ",".join([str(i) for i in h_gt]))
                out_str += "\nht:{}".format(tmp_str)
            if len(t_gh) > 0:
                if out_reverse:
                    tmp_str = "{},{},{},[{}]".format(e_map_reverse[tr[0]], e_map_reverse[gid], r_map_reverse[tr[1]], ",".join([str(e_map_reverse[i]) for i in t_gh]))
                else:
                    tmp_str = "{},{},{},[{}]".format(tr[0], gid, tr[1], ",".join([str(i) for i in t_gh]))
                out_str += "\nth:{}".format(tmp_str)
            if len(t_gt) > 0:
                if out_reverse:
                    tmp_str = "{},{},{},[{}]".format(e_map_reverse[tr[0]], e_map_reverse[gid], r_map_reverse[tr[1]], ",".join([str(e_map_reverse[i]) for i in t_gt]))
                else:
                    tmp_str = "{},{},{},[{}]".format(tr[0], gid, tr[1], ",".join([str(i) for i in t_gt]))
                out_str += "\ntt:{}".format(tmp_str)
    fout.write("{}\n".format(out_str))


def output(fout, u_id, target_items, preference, gold_ids, i2kg_map, triple_head_dict, triple_tail_dict, u_map_reverse=None, i_map_reverse=None, e_map_reverse=None, r_map_reverse=None):
    out_reverse = False
    if u_map_reverse is not None and i_map_reverse is not None and e_map_reverse is not None and r_map_reverse is not None :
        out_reverse = True
    if out_reverse:
        out_str = "user:{}\ttarget:{}".format(u_map_reverse[u_id], ",".join([e_map_reverse[i2kg_map[ti]] for ti in target_items if ti in i2kg_map]))
    else:
        out_str = "user:{}\ttarget:{}".format(u_id, ",".join([str(ti) for ti in target_items]))
    rel_ids = [preference.get((u_id, ti), -1) for ti in target_items]
    remap_tis = [i2kg_map.get(ti, -1) for ti in target_items]
    tmp_gold_ids = gold_ids - target_items
    remap_gids = [i2kg_map.get(gold_id, -1) for gold_id in tmp_gold_ids]

    for tr in zip(remap_tis, rel_ids):
        h_ids = triple_head_dict.get(tr, set())
        t_ids = triple_tail_dict.get(tr, set())
        if len(h_ids) == 0 and len(t_ids) == 0 : continue
        for gid in remap_gids:
            gh_ids = triple_head_dict.get((gid, tr[1]), set())
            gt_ids = triple_tail_dict.get((gid, tr[1]), set())
            if len(gh_ids) == 0 and len(gt_ids) == 0 : continue
            h_gh = h_ids & gh_ids
            h_gt = h_ids & gt_ids
            t_gh = t_ids & gh_ids
            t_gt = t_ids & gt_ids
            if len(h_gh) > 0:
                if out_reverse:
                    tmp_str = "{},{},{},[{}]".format(e_map_reverse[tr[0]], e_map_reverse[gid], r_map_reverse[tr[1]], ",".join([str(e_map_reverse[i]) for i in h_gh]))
                else:
                    tmp_str = "{},{},{},[{}]".format(tr[0], gid, tr[1], ",".join([str(i) for i in h_gh]))
                out_str += "\nhh:{}".format(tmp_str)
            if len(h_gt) > 0:
                if out_reverse:
                    tmp_str = "{},{},{},[{}]".format(e_map_reverse[tr[0]], e_map_reverse[gid], r_map_reverse[tr[1]], ",".join([str(e_map_reverse[i]) for i in h_gt]))
                else:
                    tmp_str = "{},{},{},[{}]".format(tr[0], gid, tr[1], ",".join([str(i) for i in h_gt]))
                out_str += "\nht:{}".format(tmp_str)
            if len(t_gh) > 0:
                if out_reverse:
                    tmp_str = "{},{},{},[{}]".format(e_map_reverse[tr[0]], e_map_reverse[gid], r_map_reverse[tr[1]], ",".join([str(e_map_reverse[i]) for i in t_gh]))
                else:
                    tmp_str = "{},{},{},[{}]".format(tr[0], gid, tr[1], ",".join([str(i) for i in t_gh]))
                out_str += "\nth:{}".format(tmp_str)
            if len(t_gt) > 0:
                if out_reverse:
                    tmp_str = "{},{},{},[{}]".format(e_map_reverse[tr[0]], e_map_reverse[gid], r_map_reverse[tr[1]], ",".join([str(e_map_reverse[i]) for i in t_gt]))
                else:
                    tmp_str = "{},{},{},[{}]".format(tr[0], gid, tr[1], ",".join([str(i) for i in t_gt]))
                out_str += "\ntt:{}".format(tmp_str)
    fout.write("{}\n".format(out_str))


model1 = 'bprmf'
model2 = 'jtransup'
root_path = '/Users/caoyixin/Github/joint-kg-recommender'
dataset_path = root_path + '/datasets/ml1m'

log1 = root_path + '/log/log/tuned_ml1m/ml1m-bprmf-analysis.log'
log2 = root_path + '/log/log/tuned_ml1m/ml1m-cjtransup-nogumbel_analysis_old.log'

users = None

u_map_file = dataset_path + '/u_map.dat'
i_map_file = dataset_path + '/i_map.dat'
e_map_file = dataset_path + '/kg/e_map.dat'
r_map_file = dataset_path + '/kg/r_map.dat'
i2kg_map_file = dataset_path + '/i2kg_map.tsv'
train_triple_file = dataset_path + '/kg/train.dat'
output_file = root_path + '/log/parse_bprmf_jtransup.log'

user_vocab, user_vocab_reverse = loadRecVocab(u_map_file)
item_vocab, item_vocab_reverse = loadRecVocab(i_map_file)
kg_vocab, kg_vocab_reverse = loadKGVocab(e_map_file)
rel_vocab, rel_vocab_reverse = loadKGVocab(r_map_file)
i2kg_map, kg2i_map = loadR2KgMap(i2kg_map_file, item_vocab=item_vocab, kg_vocab=kg_vocab)

triple_total, triple_list, triple_head_dict, triple_tail_dict = loadTriples(train_triple_file)

compareLogs(log1, log2, model1, model2, i2kg_map, triple_head_dict, triple_tail_dict, output_file, u_map_reverse=user_vocab_reverse, i_map_reverse=item_vocab_reverse, e_map_reverse=kg_vocab_reverse, r_map_reverse=rel_vocab_reverse, users=users)

