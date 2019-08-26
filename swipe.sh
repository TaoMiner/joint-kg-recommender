model_type=$1

for model_type in bprmf fm transup
do
    CUDA_VISIBLE_DEVICES=1 python run_item_recommendation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -rec_test_files valid.dat:test.dat -num_preferences 10 -l2_lambda 1e-5 -negtive_samples 1 -model_type $model_type -has_visualization -dataset ml1m -batch_size 512 -embedding_size 100 -learning_rate 0.005 -topn 10 -seed 3 -eval_interval_steps 14000 -training_steps 1400000 -early_stopping_steps_to_wait 70000
done

CUDA_VISIBLE_DEVICES=1 python run_knowledge_representation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -kg_test_files valid.dat:test.dat -l2_lambda 1e-5 -negtive_samples 1 -model_type transe -has_visualization -dataset ml1m -batch_size 512 -embedding_size 100 -learning_rate 0.005 -topn 10 -seed 3 -eval_interval_steps 1250 -training_steps 125000 -early_stopping_steps_to_wait 6250

CUDA_VISIBLE_DEVICES=1 python run_knowledge_representation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -kg_test_files valid.dat:test.dat -l2_lambda 1e-5 -negtive_samples 1 -model_type transh -has_visualization -dataset ml1m -batch_size 512 -embedding_size 100 -learning_rate 0.005 -topn 10 -seed 3 -eval_interval_steps 1250 -training_steps 125000 -early_stopping_steps_to_wait 6250

CUDA_VISIBLE_DEVICES=1 python run_knowledgable_recommendation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -rec_test_files valid.dat:test.dat -kg_test_files valid.dat:test.dat -l2_lambda 1e-5 -negtive_samples 1 -model_type jtransup -has_visualization -dataset ml1m -batch_size 512 -embedding_size 100 -learning_rate 0.005 -topn 10 -seed 3 -share_embeddings -joint_ratio 0.9 -eval_interval_steps 16000 -training_steps 1600000 -early_stopping_steps_to_wait 90000

CUDA_VISIBLE_DEVICES=1 python run_knowledgable_recommendation.py -data_path ~/joint-kg-recommender/datasets/ -log_path ~/joint-kg-recommender/log/ -rec_test_files valid.dat:test.dat -kg_test_files valid.dat:test.dat -l2_lambda 1e-5 -negtive_samples 1 -model_type jtransup -has_visualization -dataset ml1m -batch_size 512 -embedding_size 100 -learning_rate 0.005 -topn 10 -seed 3 -noshare_embeddings -joint_ratio 0.9 -eval_interval_steps 16000 -training_steps 1600000 -early_stopping_steps_to_wait 90000