import logging
import gflags
import sys
import os
import json
from tqdm import tqdm
tqdm.monitor_iterval=0
import math
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable as V

from jTransUP.models.base import get_flags, flag_defaults, init_model
from jTransUP.data.load_rating_data import load_data
from jTransUP.utils.trainer import ModelTrainer
from jTransUP.utils.misc import evalRecProcess, to_gpu, USE_CUDA
from jTransUP.utils.loss import bprLoss, orthogonalLoss, normLoss
from jTransUP.utils.visuliazer import Visualizer
from jTransUP.utils.data import getNegRatings

FLAGS = gflags.FLAGS

def evaluate(FLAGS, model, eval_iter, eval_dict, all_dicts, logger, eval_descending=True, is_report=False):

    # Evaluate
    total_batches = len(eval_iter)
    # processing bar
    pbar = tqdm(total=total_batches)
    pbar.set_description("Run Eval")

    model.eval()
    model.disable_grad()

    results = []
    for u_ids in eval_iter:
        u_var = to_gpu(V(torch.LongTensor(u_ids)))
        # batch * item
        scores = model.evaluate(u_var)
        preds = zip(u_ids, scores.data.cpu().numpy())

        results.extend( evalRecProcess(list(preds), eval_dict, all_dicts=all_dicts, descending=eval_descending, num_processes=FLAGS.num_processes, topn=FLAGS.topn, queue_limit=FLAGS.max_queue) )

        pbar.update(1)
    pbar.close()

    performances = [result[:5] for result in results]
    f1, p, r, hit, ndcg = np.array(performances).mean(axis=0)

    logger.info("f1:{:.4f}, p:{:.4f}, r:{:.4f}, hit:{:.4f}, ndcg:{:.4f}, topn:{}.".format(f1, p, r, hit, ndcg, FLAGS.topn))

    if is_report:
        predict_tuples = [result[-1] for result in results]
        for pred_tuple in predict_tuples:
            u_id = pred_tuple[0]
            top_ids = pred_tuple[1]
            gold_ids = list(pred_tuple[2])
            if FLAGS.model_type in ["transup", "jtransup", "cjtransup"]:
                for d in all_dicts:
                    gold_ids += list(d.get(u_id, set()))
                gold_ids += list(eval_dict.get(u_id, set()))
                u_var = to_gpu(V(torch.LongTensor([u_id])))
                i_var = to_gpu(V(torch.LongTensor(gold_ids)))
                # item_num * relation_total
                probs, _, _ = model.reportPreference(u_var, i_var)
                max_rel_index = torch.max(probs, 1)[1]
                gold_strs = ",".join(["{}({})".format(ir[0], ir[1]) for ir in zip(gold_ids, max_rel_index.data.tolist())])
            else:
                gold_strs = ",".join([str(i) for i in gold_ids])
            logger.info("user:{}\tgold:{}\ttop:{}".format(u_id, gold_strs, ",".join([str(i) for i in top_ids])))
    model.enable_grad()
    return f1, p, r, hit, ndcg

def train_loop(FLAGS, model, trainer, train_dataset, eval_datasets,
            user_total, item_total, logger, vis=None, is_report=False):
    train_iter, train_total, train_list, train_dict = train_dataset

    all_dicts = None
    if FLAGS.filter_wrong_corrupted:
        all_dicts = [train_dict] + [tmp_data[3] for tmp_data in eval_datasets]

    # Train.
    logger.info("Training.")

    # New Training Loop
    pbar = None
    total_loss = 0.0
    model.train()
    model.enable_grad()
    for _ in range(trainer.step, FLAGS.training_steps):

        if FLAGS.early_stopping_steps_to_wait > 0 and (trainer.step - trainer.best_step) > FLAGS.early_stopping_steps_to_wait:
            logger.info('No improvement after ' +
                       str(FLAGS.early_stopping_steps_to_wait) +
                       ' steps. Stopping training.')
            if pbar is not None: pbar.close()
            break
        if trainer.step % FLAGS.eval_interval_steps == 0 :
            if pbar is not None:
                pbar.close()
            total_loss /= FLAGS.eval_interval_steps
            logger.info("train loss:{:.4f}!".format(total_loss))

            performances = []
            for i, eval_data in enumerate(eval_datasets):
                all_eval_dicts = None
                if FLAGS.filter_wrong_corrupted:
                    all_eval_dicts = [train_dict] + [tmp_data[3] for j, tmp_data in enumerate(eval_datasets) if j!=i]
                    
                performances.append( evaluate(FLAGS, model, eval_data[0], eval_data[3], all_eval_dicts, logger, eval_descending=True if trainer.model_target == 1 else False, is_report=is_report))
            
            if trainer.step > 0 and len(performances) > 0:
                is_best = trainer.new_performance(performances[0], performances)
                
                # visuliazation
                if vis is not None:
                    vis.plot_many_stack({'Rec Train Loss': total_loss},
                    win_name="Loss Curve")
                    f1_vis_dict = {}
                    p_vis_dict = {}
                    r_vis_dict = {}
                    hit_vis_dict = {}
                    ndcg_vis_dict = {}
                    for i, performance in enumerate(performances):
                        f1_vis_dict['Rec Eval {} F1'.format(i)] = performance[0]
                        p_vis_dict['Rec Eval {} Precision'.format(i)] = performance[1]
                        r_vis_dict['Rec Eval {} Recall'.format(i)] = performance[2]
                        hit_vis_dict['Rec Eval {} Hit'.format(i)] = performance[3]
                        ndcg_vis_dict['Rec Eval {} NDCG'.format(i)] = performance[4]
                    
                    if is_best:
                        log_str = ["Best performances in {} step!".format(trainer.best_step)]
                        log_str += ["{} : {}.".format(s, "%.5f" % f1_vis_dict[s]) for s in f1_vis_dict]
                        log_str += ["{} : {}.".format(s, "%.5f" % p_vis_dict[s]) for s in p_vis_dict]
                        log_str += ["{} : {}.".format(s, "%.5f" % r_vis_dict[s]) for s in r_vis_dict]
                        log_str += ["{} : {}.".format(s, "%.5f" % hit_vis_dict[s]) for s in hit_vis_dict]
                        log_str += ["{} : {}.".format(s, "%.5f" % ndcg_vis_dict[s]) for s in ndcg_vis_dict]
                        
                        vis.log("\n".join(log_str), win_name="Best Performances")

                    vis.plot_many_stack(f1_vis_dict, win_name="Rec F1 Score@{}".format(FLAGS.topn))
                    
                    vis.plot_many_stack(p_vis_dict, win_name="Rec Precision@{}".format(FLAGS.topn))

                    vis.plot_many_stack(r_vis_dict, win_name="Rec Recall@{}".format(FLAGS.topn))

                    vis.plot_many_stack(hit_vis_dict, win_name="Rec Hit Ratio@{}".format(FLAGS.topn))

                    vis.plot_many_stack(ndcg_vis_dict, win_name="Rec NDCG@{}".format(FLAGS.topn))

            # set model in training mode
            pbar = tqdm(total=FLAGS.eval_interval_steps)
            pbar.set_description("Training")
            total_loss = 0.0
            model.train()
            model.enable_grad()

        rating_batch = next(train_iter)
        u, pi, ni = getNegRatings(rating_batch, item_total, all_dicts=all_dicts)

        u_var = to_gpu(V(torch.LongTensor(u)))
        pi_var = to_gpu(V(torch.LongTensor(pi)))
        ni_var = to_gpu(V(torch.LongTensor(ni)))

        trainer.optimizer_zero_grad()

        # Run model. output: batch_size * cand_num
        pos_score = model(u_var, pi_var)
        neg_score = model(u_var, ni_var)

        # Calculate loss.
        losses = bprLoss(pos_score, neg_score, target=trainer.model_target)
        
        if FLAGS.model_type in ["transup","transupb"]:
            user_embeddings = model.user_embeddings(u_var)
            item_embeddings = model.item_embeddings(torch.cat([pi_var, ni_var]))
            losses += orthogonalLoss(model.pref_embeddings.weight, model.pref_norm_embeddings.weight) + normLoss(user_embeddings) + normLoss(item_embeddings) + normLoss(model.pref_embeddings.weight)

        # Backward pass.
        losses.backward()

        # for param in model.parameters():
        #     print(param.grad.data.sum())

        # Hard Gradient Clipping
        nn.utils.clip_grad_norm([param for name, param in model.named_parameters()], FLAGS.clipping_max_value)

        # Gradient descent step.
        trainer.optimizer_step()
        total_loss += losses.data[0]
        pbar.update(1)

def run(only_forward=False):
    if FLAGS.seed != 0:
        random.seed(FLAGS.seed)
        torch.manual_seed(FLAGS.seed)

    # set visualization
    vis = None
    if FLAGS.has_visualization:
        vis = Visualizer(env=FLAGS.experiment_name, port=FLAGS.visualization_port)
        vis.log(json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True),
                win_name="Parameter")

    # set logger
    log_file = os.path.join(FLAGS.log_path, FLAGS.experiment_name + ".log")
    logger = logging.getLogger()
    log_level = logging.DEBUG if FLAGS.log_level == "debug" else logging.INFO
    logger.setLevel(level=log_level)
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

    logger.info("Flag Values:\n" + json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))

    # load data
    dataset_path = os.path.join(FLAGS.data_path, FLAGS.dataset)
    eval_files = FLAGS.rec_test_files.split(':')

    train_dataset, eval_datasets, u_map, i_map = load_data(dataset_path, eval_files, FLAGS.batch_size, logger=logger, negtive_samples=FLAGS.negtive_samples)

    train_iter, train_total, train_list, train_dict = train_dataset

    user_total = max(len(u_map), max(u_map.values()))
    item_total = max(len(i_map), max(i_map.values()))

    model = init_model(FLAGS, user_total, item_total, 0, 0, logger)
    epoch_length = math.ceil( train_total / FLAGS.batch_size )
    trainer = ModelTrainer(model, logger, epoch_length, FLAGS)

    if FLAGS.load_ckpt_file is not None:
        trainer.loadEmbedding(os.path.join(FLAGS.log_path, FLAGS.load_ckpt_file), model.state_dict(), cpu=not USE_CUDA)
        model.is_pretrained = True

    # Do an evaluation-only run.
    if only_forward:
        for i, eval_data in enumerate(eval_datasets):
            all_dicts = None
            if FLAGS.filter_wrong_corrupted:
                all_dicts = [train_dict] + [tmp_data[3] for j, tmp_data in enumerate(eval_datasets) if j!=i]
            evaluate(
                FLAGS,
                model,
                eval_data[0],
                eval_data[3],
                all_dicts,
                logger,
                eval_descending=True if trainer.model_target == 1 else False,
                is_report=FLAGS.is_report)
    else:
        train_loop(
            FLAGS,
            model,
            trainer,
            train_dataset,
            eval_datasets,
            user_total,
            item_total,
            logger,
            vis=vis,
            is_report=False)
    if vis is not None:
        vis.log("Finish!", win_name="Best Performances")
    
if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)
    flag_defaults(FLAGS)

    run(only_forward=FLAGS.eval_only_mode)
