from SPARQLWrapper import SPARQLWrapper, JSON
import json
import os
import time

def loadItemToKGMap(filename):
    with open(filename, 'r') as fin:
        item_to_kg_dict = {}
        for line in fin:
            line_split = line.strip().split('\t')
            if len(line_split) < 3 : continue
            item_id = line_split[0]
            db_uri = line_split[2]
            item_to_kg_dict[item_id] = db_uri
    return item_to_kg_dict

def getHeadQuery(ent):
    return "SELECT * WHERE { <%s> ?p ?o }" % ent

def getTailQuery(ent):
    return "SELECT * WHERE { ?s ?p <%s> }" % ent

def cleanHeadResults(results):
    results_cl = []
    predicate_set = set()
    entity_set = set()
    for result in results["results"]["bindings"]:
        # skip those non-eng
        if result['o']['type'] == 'literal' and 'xml:lang' in result['o'] and \
            result['o']['xml:lang'] != 'en':
            continue
        
        if result['o']['type'] == 'uri':
            entity_set.add(result['o']['value'])
        predicate_set.add(result['p']['value'])
        results_cl.append(result)
    return results_cl, predicate_set, entity_set

def cleanTailResults(results):
    results_cl = []
    predicate_set = set()
    entity_set = set()
    for result in results["results"]["bindings"]:
        entity_set.add(result['s']['value'])
        predicate_set.add(result['p']['value'])
        results_cl.append(result)
    return results_cl, predicate_set, entity_set

def downloadDBPedia(sparql, fout, entities, asTail=True):
    sec_to_wait = 60
    for ent in entities:
        print("downloading {} ...".format(ent))
        while True:
            try:
                sparql.setQuery(getHeadQuery(ent))
                head_results = sparql.query().convert()
                break
            except:
                print("http failure! wait %d seconds to retry..." % sec_to_wait)
                time.sleep(sec_to_wait)

        head_results_cl, predicate_set, entity_set = cleanHeadResults(head_results)
        head_json_str = json.dumps(head_results_cl)

        if asTail:
            while True:
                try:
                    sparql.setQuery(getTailQuery(ent))
                    tail_results = sparql.query().convert()
                    break
                except:
                    print("http failure! wait %d seconds to retry..." % sec_to_wait)
                    time.sleep(sec_to_wait)
            tail_results_cl, tail_predicate_set, tail_entity_set = cleanTailResults(tail_results)
            tail_json_str = json.dumps(tail_results_cl)
            predicate_set |= tail_predicate_set
            entity_set |= tail_entity_set
        
        fout.write(ent + '\t' + head_json_str + '\t' + tail_json_str + '\n')
        print("finish! {} entities and {} predicates!".format(len(entity_set), len(predicate_set)))
        time.sleep(1)
    return entity_set, predicate_set

if __name__ == "__main__":
    n_hop = 1

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    item2kg_file = "/home/ethan/Github/joint-kg-recommender/datasets/ml1m/MappingMovielens2DBpedia-1.2.tsv"
    kg_path = "/home/ethan/Github/joint-kg-recommender/datasets/ml1m/kg/"

    # item2kg_file = "/home/ethan/Github/joint-kg-recommender/datasets/dbbook2014/DBbook_Items_DBpedia_mapping.tsv"
    # kg_path = "/home/ethan/Github/joint-kg-recommender/datasets/dbbook2014/kg/"

    all_predicate_set = set()
    all_entity_set = set()

    item2kg_dict = loadItemToKGMap(item2kg_file)
    item_entities = set(item2kg_dict.values())

    all_entity_set.update(item_entities)
    
    input_entities = item_entities
    for i in range(n_hop):
        kg_file = os.path.join(kg_path, "kg_hop%d.dat" % i)
        with open(kg_file, 'a') as fout:
            entity_set, predicate_set = downloadDBPedia(sparql, fout, input_entities, asTail=True)
            input_entities = entity_set - all_entity_set

            all_predicate_set |= predicate_set
            all_entity_set |= entity_set
    
    predicate_file = os.path.join(kg_path, "predicate_vocab.dat")
    with open(predicate_file, 'w') as fout:
        for pred in all_predicate_set:
            fout.write(pred + '\n')
        
    entity_file = os.path.join(kg_path, "entity_vocab.dat")
    with open(entity_file, 'w') as fout:
        for ent in all_entity_set:
            fout.write(ent + '\n')