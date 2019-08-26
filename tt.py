

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

'''
train_filename = '/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/train.dat'
train_total_count, train_user_dict = loadRatings(train_filename)    
for i in range(0, 10):
    filename = '/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/test{}.dat'.format(i)

    test_total_count, test_user_dict = loadRatings(filename)
    count = 0
    for u in test_user_dict:
        count += len(train_user_dict[u])

    print(count/len(test_user_dict))
'''
train_filename = '/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/train.dat'
train_total_count, train_user_dict = loadRatings(train_filename)  
valid_filename = '/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/valid.dat'
valid_total_count, valid_user_dict = loadRatings(valid_filename)  
test_filename = '/Users/caoyixin/Github/joint-kg-recommender/datasets/ml1m/test.dat'
test_total_count, test_user_dict = loadRatings(test_filename)  

count = 0

for d in [train_user_dict, valid_user_dict, ]