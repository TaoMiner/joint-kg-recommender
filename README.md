# Unifying Knowledge Graph Learning and Recommendation Towards a Better Understanding of User Preference

This is the code of the *Unifying Knowledge Graph Learning and Recommendation Towards a Better Understanding of User Preference* in WWW'19, which proposed a model that jointly train two tasks of item recommendation and KG representation learning.

## Environment

python 3.6

Pytorch 0.3.x

visdom if visualization flag is set to True.

some required packages are included in *requirements.txt*.

## Run our codes

We implement the models including our proposed TUP and KTUP as well as some baselines: BPRMF, FM, CFKG, CKE, CoFM, TransE, TransH and TransR. We split them into three types: Item recommendation, knowledge representation and the joint model of two tasks, which correspond to run_item_recommendation.py, run_knowledge_representation.py and run_knowledgable_recommendation.py, respectively. Each model has an example shell file to run the code.

Take item recommendation for example, to run each model, simply:

`python run_item_recommendation.py -model_type REC_MODEL -dataset DATASET_NAME -data_path PATH_TO_DATASET_FOLDER -log_path PATH_TO_LOG -rec_test_files EVAL_FILENAMES -nohas_visualization`

For knowledge representation, simply:

`python run_knowledge_representation.py -model_type KG_MODEL -dataset DATASET_NAME -data_path PATH_TO_DATASET_FOLDER -log_path PATH_TO_LOG -kg_test_files EVAL_FILENAMES -nohas_visualization`

For joint model, simplY:

`python run_knowledgable_recommendation.py -model_type JOINT_MODEL -dataset DATASET_NAME -data_path PATH_TO_DATASET_FOLDER -log_path PATH_TO_LOG -rec_test_files REC_EVAL_FILENAMES -kg_test_files KG_EVAL_FILENAMES -nohas_visualization`

we now describe the main flags: datasets, models and visualization:

## Datasets

We use two datasets: movielens-1m (ml1m for short) and dbbook2014. We collect the related facts from DBPedia, where the triplets are directly related to the entities with mapped items, no matter which role (i.e. subject or object) the entity serves as. The processed datasets can be download [here](https://drive.google.com/file/d/1FIbaWzP6AWUNG2-8q6SKQ3b9yTiiLvGW/view?usp=sharing), and the original data is limited to their authority.

The flag '-data_path' is used to specify the root path to dataset.

The flag '-dataset' is used to specify the dataset, 'ml1m' for movielens-1m, 'dbbook2014' for dbbook2014, and any dataset folder names under the data path.

Thus, all dataset related files shall be put into the folder '/PATH_TO_DATASET_FOLDER/DATASET_NAME/'.

Now, we detail the required files.

For item recommendation, the folder should contain the following files: **train.dat**, **u_map.dat**, **i_map.dat**, where each line in train.data is a triple: 'user_id item_id rating' that separated by '\t', u_map and i_map specify the mapped user_id to original user_id. The evaluation files would be specified by flag '-rec_test_files', where multiple eval files separated by ':'. Note that the first eval file is used for validation.

For KG representation, the files should under the path: '/PATH_TO_DATASET_FOLDER/DATASET_NAME/kg/'. The required files contain: **train.dat**, **e_map.dat** and **r_map.dat**. Similarly, each line in train.dat is a triple: 'head_entity_id tail_entity_id relation_id' separated by '\t', e_map and r_map specify the mapped entity_id to original entity. The evaluation files would be specified by flag '-kg_test_files', where multiple eval files separated by ':'. Note that the first eval file is used for validation.

The joint model requires all of the above files and **i2kg_map.tsv**, where each line consist of original item id, entity title, and original entity uri separated by tab.

For example, we run our KTUP by:

`python run_knowledgable_recommendation.py -model_type jtransup -dataset ml1m -data_path ~/joint-kg-recommender/datasets/ -log_path ~/Github/joint-kg-recommender/log/ -rec_test_files valid.dat:test.dat -kg_test_files valid.dat:test.dat -nohas_visualization`

Then, we need a folder '~/joint-kg-recommender/datasets/ml1m/' including the required files as:

```
ml1m
│   train.dat
│   valid.dat
│   test.dat
│   u_map.dat
│   i_map.dat
│   i2kg_map.tsv 
│
└───kg
│   │   train.dat
│   │   valid.dat
│   │   test.dat
│   │   e_map.dat
│   │   r_map.dat
```


## Models

We use the flag '-model_type' to specify the model used, which has to be chosen from the following models: ['bpr','fm','trasne','transh','transr','cfkg','cke','cofm','transup','jtransup'].

Specifically, ['bpr','fm','transup'] is for item recommendation. 'transup' is our proposed **TUP**. ['trasne','trasnh','trasnr'] is for kg representation. ['cfkg','cke','cofm','jtransup'] is for item recommendation. 'jtransup' is our proposed **KTUP**.

### Model Specific Flags

For TUP, there are two specific flags: '-num_preferences' and '-use_st_gumbel', which denotes the number of user preferences and if we use hard strategy for preference induction, respectively.

For joint models, there are also two flags: '-joint_ratio' and '-share_embeddings', which denote the ratio of training data in each batch between item recommendation and KG representation, and if the two tasks share embeddings of aligned items and entities. Note that for model 'cfkg', it must be '-share_embeddings' due to the its basic idea of unified graph of items and entities. For models 'cke' and 'jtransup' (KTUP), it must be '-noshare_embeddings'.

### General Flags

We can also specify the general parameters by setting flags like optimizer or learning rate, which can be found in './models/base.py'.

### Visualization

We use the package of visdom for visualization. If you decide to visualize the training and evaluation curve, the visdom environment is required (python -m visdom.server) and set '-has_visualization', and even the port '-visualization_port 8097'. Then, one can moniter the training and evaluation curves using the brower by entering : "http://host_ip:8097".

## Reference
If you use our code, please cite our paper:
```
@inproceedings{cao2018unifying,
  title={Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preference},
  author={Cao, Yixin and Wang, Xiang and He, Xiangnan and Hu, Zikun and Chua Tat-seng},
  booktitle={WWW},
  year={2019}
}
```