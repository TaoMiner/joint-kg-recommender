from jTransUP.data.preprocessRatings import preprocess as preprocessRating
from jTransUP.data.preprocessTriples import preprocess as preprocessKG
import os
import logging

data_path = "/Users/caoyixin/Github/joint-kg-recommender/datasets/"
dataset = 'dbbook2014'

dataset_path = os.path.join(data_path, dataset)
kg_path = os.path.join(dataset_path, 'kg')

rating_file = os.path.join(dataset_path, 'ratings.csv')
triple_file = os.path.join(kg_path, "kg_hop0.dat")
relation_file = os.path.join(kg_path, "relation_filter.dat")
i2kg_file = os.path.join(dataset_path, "i2kg_map.tsv")

log_path = dataset_path

logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)

log_file = os.path.join(dataset_path, "data_preprocess.log")
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

preprocessRating(rating_file, dataset_path, low_frequence=5, logger=logger)

preprocessKG([triple_file], kg_path, entity_file=i2kg_file, relation_file=relation_file, logger=logger)
