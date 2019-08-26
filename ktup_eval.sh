rec_test_files=$1

for rec_test_files in test0.dat test1.dat test2.dat test3.dat test4.dat
do
    CUDA_VISIBLE_DEVICES=0 python run_knowledgable_recommendation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -rec_test_files $rec_test_files -l2_lambda 0 -model_type cjtransup -has_visualization -dataset ml1m -batch_size 400 -embedding_size 100 -learning_rate 0.001 -topn 10 -seed 3 -eval_interval_steps 19520 -training_steps 1952000 -early_stopping_steps_to_wait 97600 -optimizer_type Adam -joint_ratio 0.7 -noshare_embeddings -L1_flag -norm_lambda 1 -kg_lambda 1 -nouse_st_gumbel -visualization_port 8098 -eval_only_mode -is_report -load_ckpt_file
done

kg_test_files=$2
for kg_test_files in one2one.dat one2N.dat N2one.dat N2N.dat
do
    CUDA_VISIBLE_DEVICES=0 python run_knowledgable_recommendation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -kg_test_files test0.dat -kg_test_files  -l2_lambda 0 -model_type cjtransup -has_visualization -dataset dbbook2014 -batch_size 400 -embedding_size 100 -learning_rate 0.001 -topn 10 -seed 3 -eval_interval_steps 19520 -training_steps 1952000 -early_stopping_steps_to_wait 97600 -optimizer_type Adam -joint_ratio 0.7 -noshare_embeddings -L1_flag -norm_lambda 1 -kg_lambda 1 -nouse_st_gumbel -visualization_port 8098 -eval_only_mode -is_report -load_ckpt_file
done