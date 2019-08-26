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
from jTransUP.data.load_triple_data import load_data
from jTransUP.utils.trainer import ModelTrainer
from jTransUP.utils.misc import to_gpu, evalKGProcess, USE_CUDA
from jTransUP.utils.loss import bprLoss, orthogonalLoss, normLoss
from jTransUP.utils.visuliazer import Visualizer
from jTransUP.utils.data import getTrainTripleBatch
import jTransUP.utils.loss as loss

FLAGS = gflags.FLAGS

def evaluate(FLAGS, model, entity_total, relation_total, eval_head_iter, eval_tail_iter, eval_head_dict, eval_tail_dict, all_head_dicts, all_tail_dicts, logger, eval_descending=True, is_report=False):
    # Evaluate
    total_batches = len(eval_head_iter) + len(eval_tail_iter)
    # processing bar
    pbar = tqdm(total=total_batches)
    pbar.set_description("Run Eval")

    model.eval()
    model.disable_grad()

    # head prediction evaluation
    head_results = []
    for batch_trs in eval_head_iter:
        t = [tr[0] for tr in batch_trs]
        r = [tr[1] for tr in batch_trs]
        t_var = to_gpu(V(torch.LongTensor(t)))
        r_var = to_gpu(V(torch.LongTensor(r)))
        # batch * item
        scores = model.evaluateHead(t_var, r_var)
        preds = zip(batch_trs, scores.data.cpu().numpy())
        
        head_results.extend( evalKGProcess(list(preds), eval_head_dict, all_dicts=all_head_dicts, descending=eval_descending, num_processes=FLAGS.num_processes, topn=FLAGS.topn, queue_limit=FLAGS.max_queue) )

        pbar.update(1)
    # head prediction evaluation
    tail_results = []
    for batch_hrs in eval_tail_iter:
        h = [hr[0] for hr in batch_hrs]
        r = [hr[1] for hr in batch_hrs]
        h_var = to_gpu(V(torch.LongTensor(h)))
        r_var = to_gpu(V(torch.LongTensor(r)))
        # batch * item
        scores = model.evaluateTail(h_var, r_var)
        preds = zip(batch_hrs, scores.data.cpu().numpy())

        tail_results.extend( evalKGProcess(list(preds), eval_tail_dict, all_dicts=all_tail_dicts, descending=eval_descending, num_processes=FLAGS.num_processes, topn=FLAGS.topn, queue_limit=FLAGS.max_queue) )

        pbar.update(1)

    pbar.close()

    # hit, rank
    head_performances = [result[:2] for result in head_results]
    tail_performances = [result[:2] for result in tail_results]

    head_hit, head_mean_rank = np.array(head_performances).mean(axis=0)

    tail_hit, tail_mean_rank = np.array(tail_performances).mean(axis=0)

    logger.info("head hit:{:.4f}, head mean rank:{:.4f}, topn:{}.".format(head_hit, head_mean_rank, FLAGS.topn))

    logger.info("tail hit:{:.4f}, tail mean rank:{:.4f}, topn:{}.".format(tail_hit, tail_mean_rank, FLAGS.topn))

    head_num = len(head_results)
    tail_num = len(tail_results)

    avg_hit = float(head_hit * head_num + tail_hit * tail_num) / (head_num + tail_num)
    avg_mean_rank = float(head_mean_rank * head_num + tail_mean_rank * tail_num) / (head_num + tail_num)

    logger.info("avg hit:{:.4f}, avg mean rank:{:.4f}, topn:{}.".format(avg_hit, avg_mean_rank, FLAGS.topn))

    if is_report:
        for result in head_results:
            hit = result[0]
            rank = result[1]
            t = result[2][0]
            r = result[2][1]
            gold_h = result[3]
            logger.info("H\t{}\t{}\t{}\t{}".format(gold_h, t, r, hit))
        for result in tail_results:
            hit = result[0]
            rank = result[1]
            h = result[2][0]
            r = result[2][1]
            gold_t = result[3]
            logger.info("T\t{}\t{}\t{}\t{}".format(h, gold_t, r, hit))
    model.enable_grad()
    return avg_hit, avg_mean_rank

def train_loop(FLAGS, model, trainer, train_dataset, eval_datasets,
            entity_total, relation_total, logger, vis=None, is_report=False):
    train_iter, train_total, train_list, train_head_dict, train_tail_dict = train_dataset

    all_head_dicts = None
    all_tail_dicts = None
    if FLAGS.filter_wrong_corrupted:
        all_head_dicts = [train_head_dict] + [tmp_data[4] for tmp_data in eval_datasets]
        all_tail_dicts = [train_tail_dict] + [tmp_data[5] for tmp_data in eval_datasets]

    # Train.
    logger.info("Training.")

    # New Training Loop
    pbar = None
    total_loss = 0.0
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
                eval_head_dicts = None
                eval_tail_dicts = None
                if FLAGS.filter_wrong_corrupted:
                    eval_head_dicts = [train_head_dict] + [tmp_data[4] for j, tmp_data in enumerate(eval_datasets) if j!=i]
                    eval_tail_dicts = [train_tail_dict] + [tmp_data[5] for j, tmp_data in enumerate(eval_datasets) if j!=i]

                performances.append( evaluate(FLAGS, model, entity_total, relation_total, eval_data[0], eval_data[1], eval_data[4], eval_data[5], eval_head_dicts, eval_tail_dicts, logger, eval_descending=False, is_report=is_report))

            if trainer.step > 0 and len(performances) > 0 :
                is_best = trainer.new_performance(performances[0], performances)
                # visuliazation
                if vis is not None:
                    vis.plot_many_stack({'KG Train Loss': total_loss},
                    win_name="Loss Curve")
                    hit_vis_dict = {}
                    meanrank_vis_dict = {}
                    for i, performance in enumerate(performances):
                        hit_vis_dict['KG Eval {} Hit'.format(i)] = performance[0]
                        meanrank_vis_dict['KG Eval {} MeanRank'.format(i)] = performance[1]
                    
                    if is_best:
                        log_str = ["Best performances in {} step!".format(trainer.best_step)]
                        log_str += ["{} : {}.".format(s, "%.5f" % hit_vis_dict[s]) for s in hit_vis_dict]
                        log_str += ["{} : {}.".format(s, "%.5f" % meanrank_vis_dict[s]) for s in meanrank_vis_dict]
                        vis.log("\n".join(log_str), win_name="Best Performances")

                    vis.plot_many_stack(hit_vis_dict, win_name="KG Hit Ratio@{}".format(FLAGS.topn))

                    vis.plot_many_stack(meanrank_vis_dict, win_name="KG MeanRank")
            # set model in training mode
            pbar = tqdm(total=FLAGS.eval_interval_steps)
            pbar.set_description("Training")
            total_loss = 0.0
            model.train()
            model.enable_grad()

        triple_batch = next(train_iter)
        ph, pt, pr, nh, nt, nr = getTrainTripleBatch(triple_batch, entity_total, all_head_dicts=all_head_dicts, all_tail_dicts=all_tail_dicts)

        ph_var = to_gpu(V(torch.LongTensor(ph)))
        pt_var = to_gpu(V(torch.LongTensor(pt)))
        pr_var = to_gpu(V(torch.LongTensor(pr)))
        nh_var = to_gpu(V(torch.LongTensor(nh)))
        nt_var = to_gpu(V(torch.LongTensor(nt)))
        nr_var = to_gpu(V(torch.LongTensor(nr)))

        trainer.optimizer_zero_grad()

        # Run model. output: batch_size * 1
        pos_score = model(ph_var, pt_var, pr_var)
        neg_score = model(nh_var, nt_var, nr_var)

        # Calculate loss.
        # losses = nn.MarginRankingLoss(margin=FLAGS.margin).forward(pos_score, neg_score, to_gpu(torch.autograd.Variable(torch.FloatTensor([trainer.model_target]*len(ph)))))

        losses = loss.marginLoss()(pos_score, neg_score, FLAGS.margin)
        
        ent_embeddings = model.ent_embeddings(torch.cat([ph_var, pt_var, nh_var, nt_var]))
        rel_embeddings = model.rel_embeddings(torch.cat([pr_var, nr_var]))
        
        if FLAGS.model_type == "transh":
            norm_embeddings = model.norm_embeddings(torch.cat([pr_var, nr_var]))
            losses += loss.orthogonalLoss(rel_embeddings, norm_embeddings)

        losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings)
        
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
    trainer.save(trainer.checkpoint_path + '_final')

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
    kg_path = os.path.join(os.path.join(FLAGS.data_path, FLAGS.dataset), 'kg')
    eval_files = []
    if FLAGS.kg_test_files:
        eval_files = FLAGS.kg_test_files.split(':')

    train_dataset, eval_datasets, e_map, r_map = load_data(kg_path, eval_files, FLAGS.batch_size, logger=logger, negtive_samples=FLAGS.negtive_samples)

    entity_total = max(len(e_map), max(e_map.values()))
    relation_total = max(len(r_map), max(r_map.values()))
    
    train_iter, train_total, train_list, train_head_dict, train_tail_dict = train_dataset

    model = init_model(FLAGS, 0, 0, entity_total, relation_total, logger)
    epoch_length = math.ceil( train_total / FLAGS.batch_size )
    trainer = ModelTrainer(model, logger, epoch_length, FLAGS)

    # todo : load ckpt full path
    if FLAGS.load_ckpt_file is not None:
        trainer.loadEmbedding(os.path.join(FLAGS.log_path, FLAGS.load_ckpt_file), model.state_dict(), cpu=not USE_CUDA)
        model.is_pretrained = True

    # Do an evaluation-only run.
    if only_forward:
        # head_iter, tail_iter, eval_total, eval_list, eval_head_dict, eval_tail_dict
        for i, eval_data in enumerate(eval_datasets):
            all_head_dicts = None
            all_tail_dicts = None
            if FLAGS.filter_wrong_corrupted:
                all_head_dicts = [train_head_dict] + [tmp_data[4] for j, tmp_data in enumerate(eval_datasets) if j!=i]
                all_tail_dicts = [train_tail_dict] + [tmp_data[5] for j, tmp_data in enumerate(eval_datasets) if j!=i]
            evaluate(
                FLAGS,
                model,
                entity_total,
                relation_total,
                eval_data[0],
                eval_data[1],
                eval_data[4],
                eval_data[5],
                all_head_dicts,
                all_tail_dicts,
                logger,
                eval_descending=False,
                is_report=FLAGS.is_report)
    else:
        train_loop(
            FLAGS,
            model,
            trainer,
            train_dataset,
            eval_datasets,
            entity_total,
            relation_total,
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
