from jTransUP.data.load_rating_data import load_data
from jTransUP.utils.data import MakeTrainIterator, MakeEvalIterator
import os

trainDict, testDict, validDict, allRatingDict, user_total, item_total, trainTotal, testTotal, validTotal = load_data("/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/")
print("user:{}, item:{}!".format(user_total, item_total))
print("totally ratings for {} train, {} valid, and {} test!".format(trainTotal, validTotal, testTotal))
u_id = 249
print("u:{} has brought items for train {}, valid {} and test {}!".format(u_id, trainDict[u_id], validDict[u_id] if validDict is not None else [], testDict[u_id]))

eval_iter = MakeEvalIterator(validDict, item_total, 100, allRatingDict=allRatingDict)
eval_total = 0
item_count = 0
while True:
    rating_batch = next(eval_iter)
    if rating_batch is None: break
    u, pi = rating_batch
    for i in u:
        if i == 249: item_count += 1
    eval_total += len(u)
print(eval_total)
print("user {} has {} eval!".format(u_id, item_count))
