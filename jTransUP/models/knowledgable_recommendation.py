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
from jTransUP.data.load_kg_rating_data import load_data
from jTransUP.utils.trainer import ModelTrainer
from jTransUP.utils.misc import evalRecProcess, evalKGProcess, to_gpu
from jTransUP.utils.loss import bprLoss, orthogonalLoss, normLoss
from jTransUP.utils.visuliazer import Visualizer
from jTransUP.utils.data import getTrainTripleBatch, getNegRatings
import jTransUP.utils.loss as loss

FLAGS = gflags.FLAGS

def getMappedEntities(i_ids, i_remap, new_map):
    e_ids = []
    new_i_ids = []
    for i in set(i_ids):
        if i not in i_remap : continue
        new_index = new_map[i_remap[i]]
        if new_index[0] == -1: continue
        e_ids.append(new_index[0])
        new_i_ids.append(new_index[1])
    return e_ids, new_i_ids

def getMappedItems(e_ids, e_remap, new_map):
    i_ids = []
    new_e_ids = []
    for e in set(e_ids):
        if e not in e_remap : continue
        new_index = new_map[e_remap[e]]
        if new_index[1] == -1: continue
        i_ids.append(new_index[1])
        new_e_ids.append(new_index[0])
    return new_e_ids, i_ids

def evaluateRec(FLAGS, model, eval_iter, eval_dict, all_dicts, i_map, logger, eval_descending=True, is_report=False):
    # Evaluate
    total_batches = len(eval_iter)
    # processing bar
    pbar = tqdm(total=total_batches)
    pbar.set_description("Run Eval")

    all_i_var = None
    if FLAGS.share_embeddings:
        all_i_ids = [i_map[i] for i in range(len(i_map))]
        all_i_var = to_gpu(V(torch.LongTensor(all_i_ids)))

    model.eval()
    model.disable_grad()

    results = []
    for u_ids in eval_iter:
        u_var = to_gpu(V(torch.LongTensor(u_ids)))
        # batch * item
        scores = model.evaluateRec(u_var, all_i_ids=all_i_var)
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
            if FLAGS.model_type in ["transup", "jtransup"]:
                for d in all_dicts:
                    gold_ids += list(d.get(u_id, set()))
                gold_ids += list(eval_dict.get(u_id, set()))
                
                u_var = to_gpu(V(torch.LongTensor([u_id])))
                i_var = to_gpu(V(torch.LongTensor(gold_ids)))

                probs, _, _ = model.reportPreference(u_var, i_var)
                max_rel_index = torch.max(probs, 1)[1]
                gold_strs = ",".join(["{}({})".format(ir[0], ir[1]) for ir in zip(gold_ids, max_rel_index.data.tolist())])
            else:
                gold_strs = ",".join([str(i) for i in gold_ids])
            logger.info("user:{}\tgold:{}\ttop:{}".format(u_id, gold_strs, ",".join([str(i) for i in top_ids])))
    model.enable_grad()
    return f1, p, r, hit, ndcg

def evaluateKG(FLAGS, model, eval_head_iter, eval_tail_iter, eval_head_dict, eval_tail_dict, all_head_dicts, all_tail_dicts, e_map, logger, eval_descending=True, is_report=False):
    # Evaluate
    total_batches = len(eval_head_iter) + len(eval_tail_iter)
    # processing bar
    pbar = tqdm(total=total_batches)
    pbar.set_description("Run Eval")

    model.eval()
    model.disable_grad()

    all_e_var = None
    if FLAGS.share_embeddings:
        all_e_ids = [e_map[e] for e in range(len(e_map))]
        all_e_var = to_gpu(V(torch.LongTensor(all_e_ids)))

    # head prediction evaluation
    head_results = []
    
    for batch_trs in eval_head_iter:
        t = [tr[0] for tr in batch_trs] if FLAGS.share_embeddings else [e_map[tr[0]] for tr in batch_trs]
        r = [tr[1] for tr in batch_trs]
        t_var = to_gpu(V(torch.LongTensor(t)))
        r_var = to_gpu(V(torch.LongTensor(r)))
        # batch * item
        scores = model.evaluateHead(t_var, r_var, all_e_ids=all_e_var)
        preds = zip(batch_trs, scores.data.cpu().numpy())
        
        head_results.extend( evalKGProcess(list(preds), eval_head_dict, all_dicts=all_head_dicts, descending=eval_descending, num_processes=FLAGS.num_processes, topn=FLAGS.topn, queue_limit=FLAGS.max_queue) )

        pbar.update(1)
    
    # tail prediction evaluation
    tail_results = []
    for batch_hrs in eval_tail_iter:
        h = [hr[0] for hr in batch_hrs] if FLAGS.share_embeddings else [e_map[hr[0]] for hr in batch_hrs]
        r = [hr[1] for hr in batch_hrs]
        h_var = to_gpu(V(torch.LongTensor(h)))
        r_var = to_gpu(V(torch.LongTensor(r)))
        # batch * item
        scores = model.evaluateTail(h_var, r_var, all_e_ids=all_e_var)
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

def train_loop(FLAGS, model, trainer, rating_train_dataset, triple_train_dataset, rating_eval_datasets, triple_eval_datasets, e_map, i_map, ikg_map, logger, vis=None, is_report=False):
    rating_train_iter, rating_train_total, rating_train_list, rating_train_dict = rating_train_dataset

    triple_train_iter, triple_train_total, triple_train_list, head_train_dict, tail_train_dict = triple_train_dataset

    all_rating_dicts = None
    if FLAGS.filter_wrong_corrupted:
        all_rating_dicts = [rating_train_dict] + [tmp_data[3] for tmp_data in rating_eval_datasets]
    
    all_head_dicts = None
    all_tail_dicts = None
    if FLAGS.filter_wrong_corrupted:
        all_head_dicts = [head_train_dict] + [tmp_data[4] for tmp_data in triple_eval_datasets]
        all_tail_dicts = [tail_train_dict] + [tmp_data[5] for tmp_data in triple_eval_datasets]

    item_total = len(i_map)
    entity_total = len(e_map)
    step_to_switch = 10 * FLAGS.joint_ratio

    # Train.
    logger.info("Training.")

    # New Training Loop
    pbar = None
    rec_total_loss = 0.0
    kg_total_loss = 0.0
    model.train()
    model.enable_grad()
    for _ in range(trainer.step, FLAGS.training_steps):

        if FLAGS.early_stopping_steps_to_wait > 0 and (trainer.step - trainer.best_step) > FLAGS.early_stopping_steps_to_wait:
            logger.info('No improvement after ' +
                       str(FLAGS.early_stopping_steps_to_wait) +
                       ' steps. Stopping training.')
            if pbar is not None: pbar.close()
            break
        if trainer.step % FLAGS.eval_interval_steps == 0 and (len(rating_eval_datasets) > 0 or len(triple_eval_datasets)>0):
            if pbar is not None:
                pbar.close()
            rec_total_loss /= (FLAGS.eval_interval_steps * FLAGS.joint_ratio)
            kg_total_loss /= (FLAGS.eval_interval_steps * (1-FLAGS.joint_ratio))
            logger.info("rec train loss:{:.4f}, kg train loss:{:.4f}!".format(rec_total_loss, kg_total_loss))

            rec_performances = []
            for i, eval_data in enumerate(rating_eval_datasets):
                all_eval_dicts = None
                if FLAGS.filter_wrong_corrupted:
                    all_eval_dicts = [rating_train_dict] + [tmp_data[3] for j, tmp_data in enumerate(rating_eval_datasets) if j!=i]

                rec_performances.append( evaluateRec(FLAGS, model, eval_data[0], eval_data[3], all_eval_dicts, i_map, logger, eval_descending=True if trainer.model_target == 1 else False, is_report=is_report))

            kg_performances = []
            for i, eval_data in enumerate(triple_eval_datasets):
                eval_head_dicts = None
                eval_tail_dicts = None
                if FLAGS.filter_wrong_corrupted:
                    eval_head_dicts = [head_train_dict] + [tmp_data[4] for j, tmp_data in enumerate(triple_eval_datasets) if j!=i]
                    eval_tail_dicts = [tail_train_dict] + [tmp_data[5] for j, tmp_data in enumerate(triple_eval_datasets) if j!=i]

                kg_performances.append( evaluateKG(FLAGS, model, eval_data[0], eval_data[1], eval_data[4], eval_data[5], eval_head_dicts, eval_tail_dicts, e_map, logger, eval_descending=False, is_report=is_report))

            vis_rec = True if len(rec_performances) > 0 else False
            vis_kg = True if len(kg_performances) > 0 else False
            if trainer.step > 0 and vis_rec > 0:
                is_best = trainer.new_performance(rec_performances[0], rec_performances)
                # visuliazation
                if vis is not None:
                    if vis_rec and vis_kg:
                        vis.plot_many_stack({'Rec Train Loss': rec_total_loss, 'KG Train Loss':kg_total_loss}, win_name="Loss Curve")
                    else:
                        vis.plot_many_stack({'Rec Train Loss': rec_total_loss}, win_name="Loss Curve")

                    f1_dict = {}
                    p_dict = {}
                    r_dict = {}
                    rec_hit_dict = {}
                    ndcg_dict = {}
                    for i, performance in enumerate(rec_performances):
                        f1_dict['Rec Eval {} F1'.format(i)] = performance[0]
                        p_dict['Rec Eval {} Precision'.format(i)] = performance[1]
                        r_dict['Rec Eval {} Recall'.format(i)] = performance[2]
                        rec_hit_dict['Rec Eval {} Hit'.format(i)] = performance[3]
                        ndcg_dict['Rec Eval {} NDCG'.format(i)] = performance[4]
                    
                    kg_hit_dict = {}
                    meanrank_dict = {}
                    for i, performance in enumerate(kg_performances):
                        kg_hit_dict['KG Eval {} Hit'.format(i)] = performance[0]
                        meanrank_dict['KG Eval {} MeanRank'.format(i)] = performance[1]

                    if is_best:
                        log_str = ["Best performances in {} step!".format(trainer.best_step)]
                        log_str += ["{} : {}.".format(s, "%.5f" % f1_dict[s]) for s in f1_dict]
                        log_str += ["{} : {}.".format(s, "%.5f" % p_dict[s]) for s in p_dict]
                        log_str += ["{} : {}.".format(s, "%.5f" % r_dict[s]) for s in r_dict]
                        log_str += ["{} : {}.".format(s, "%.5f" % rec_hit_dict[s]) for s in rec_hit_dict]
                        log_str += ["{} : {}.".format(s, "%.5f" % ndcg_dict[s]) for s in ndcg_dict]

                        if vis_kg:
                            log_str += ["{} : {}.".format(s, "%.5f" % kg_hit_dict[s]) for s in kg_hit_dict]
                            log_str += ["{} : {}.".format(s, "%.5f" % meanrank_dict[s]) for s in meanrank_dict]
                        
                        vis.log("\n".join(log_str), win_name="Best Performances")

                    vis.plot_many_stack(f1_dict, win_name="Rec F1 Score@{}".format(FLAGS.topn))
                    
                    vis.plot_many_stack(p_dict, win_name="Rec Precision@{}".format(FLAGS.topn))

                    vis.plot_many_stack(r_dict, win_name="Rec Recall@{}".format(FLAGS.topn))

                    vis.plot_many_stack(rec_hit_dict, win_name="Rec Hit Ratio@{}".format(FLAGS.topn))

                    vis.plot_many_stack(ndcg_dict, win_name="Rec NDCG@{}".format(FLAGS.topn))
                    if vis_kg:
                        vis.plot_many_stack(kg_hit_dict, win_name="KG Hit Ratio@{}".format(FLAGS.topn))

                        vis.plot_many_stack(meanrank_dict, win_name="KG MeanRank")

            # set model in training mode
            pbar = tqdm(total=FLAGS.eval_interval_steps)
            pbar.set_description("Training")
            rec_total_loss = 0.0
            kg_total_loss = 0.0
    
            model.train()
            model.enable_grad()

        # recommendation train
        if trainer.step % 10 < step_to_switch :
            rating_batch = next(rating_train_iter)
            u, pi, ni = getNegRatings(rating_batch, item_total, all_dicts=all_rating_dicts)

            e_ids, i_ids = getMappedEntities(pi+ni, i_map, ikg_map)

            if FLAGS.share_embeddings:
                ni = [i_map[i] for i in ni]
                pi = [i_map[i] for i in pi]

            u_var = to_gpu(V(torch.LongTensor(u)))
            pi_var = to_gpu(V(torch.LongTensor(pi)))
            ni_var = to_gpu(V(torch.LongTensor(ni)))

            trainer.optimizer_zero_grad()

            # Run model. output: batch_size * cand_num, input: ratings, triples, is_rec=True
            pos_score = model( (u_var, pi_var), None, is_rec=True)
            neg_score = model( (u_var, ni_var), None, is_rec=True)

            # Calculate loss.
            losses = bprLoss(pos_score, neg_score, target=trainer.model_target)
            
            if FLAGS.model_type in ["transup", "jtransup"]:
                losses += orthogonalLoss(model.pref_embeddings.weight, model.pref_norm_embeddings.weight)
        # kg train
        else :
            triple_batch = next(triple_train_iter)
            ph, pt, pr, nh, nt, nr = getTrainTripleBatch(triple_batch, entity_total, all_head_dicts=all_head_dicts, all_tail_dicts=all_tail_dicts)

            e_ids, i_ids = getMappedItems(ph+pt+nh+nt, e_map, ikg_map)

            if FLAGS.share_embeddings:
                ph = [e_map[e] for e in ph]
                pt = [e_map[e] for e in pt]
                nh = [e_map[e] for e in nh]
                nt = [e_map[e] for e in nt]

            ph_var = to_gpu(V(torch.LongTensor(ph)))
            pt_var = to_gpu(V(torch.LongTensor(pt)))
            pr_var = to_gpu(V(torch.LongTensor(pr)))
            nh_var = to_gpu(V(torch.LongTensor(nh)))
            nt_var = to_gpu(V(torch.LongTensor(nt)))
            nr_var = to_gpu(V(torch.LongTensor(nr)))

            trainer.optimizer_zero_grad()

            # Run model. output: batch_size * cand_nu, input: ratings, triples, is_rec=True
            pos_score = model(None, (ph_var, pt_var, pr_var), is_rec=False)
            neg_score = model(None, (nh_var, nt_var, nr_var), is_rec=False)

            # Calculate loss.
            # losses = nn.MarginRankingLoss(margin=FLAGS.margin).forward(pos_score, neg_score, to_gpu(torch.autograd.Variable(torch.FloatTensor([trainer.model_target]*len(ph)))))

            losses = loss.marginLoss()(pos_score, neg_score, FLAGS.margin)
            
            ent_embeddings = model.ent_embeddings(torch.cat([ph_var, pt_var, nh_var, nt_var]))
            rel_embeddings = model.rel_embeddings(torch.cat([pr_var, nr_var]))
            if FLAGS.model_type in ["jtransup"]:
                norm_embeddings = model.norm_embeddings(torch.cat([pr_var, nr_var]))
                losses += loss.orthogonalLoss(rel_embeddings, norm_embeddings)

            losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings)
            losses = FLAGS.kg_lambda * losses
        # align loss if not share embeddings
        if not FLAGS.share_embeddings and FLAGS.model_type not in ['cke', 'jtransup']:
            e_var = to_gpu(V(torch.LongTensor(e_ids)))
            i_var = to_gpu(V(torch.LongTensor(i_ids)))
            ent_embeddings = model.ent_embeddings(e_var)
            item_embeddings = model.item_embeddings(i_var)
            losses += FLAGS.norm_lambda * loss.pNormLoss(ent_embeddings, item_embeddings, L1_flag=FLAGS.L1_flag)
        

        # Backward pass.
        losses.backward()

        # for param in model.parameters():
        #     print(param.grad.data.sum())

        # Hard Gradient Clipping
        nn.utils.clip_grad_norm([param for name, param in model.named_parameters()], FLAGS.clipping_max_value)

        # Gradient descent step.
        trainer.optimizer_step()
        if trainer.step % 10 < step_to_switch :
            rec_total_loss += losses.data[0]
        else:
            kg_total_loss += losses.data[0]
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
    rec_eval_files = []
    kg_eval_files = []
    if FLAGS.rec_test_files is not None:
        rec_eval_files = FLAGS.rec_test_files.split(':')
    if FLAGS.kg_test_files is not None:
        kg_eval_files = FLAGS.kg_test_files.split(':')

    rating_train_dataset, rating_eval_datasets, u_map, i_map, triple_train_dataset, triple_eval_datasets, e_map, r_map, ikg_map = load_data(dataset_path, rec_eval_files, kg_eval_files, FLAGS.batch_size, negtive_samples=FLAGS.negtive_samples, logger=logger)

    rating_train_iter, rating_train_total, rating_train_list, rating_train_dict = rating_train_dataset

    triple_train_iter, triple_train_total, triple_train_list, triple_train_head_dict, triple_train_tail_dict = triple_train_dataset

    user_total = max(len(u_map), max(u_map.values()))
    item_total = max(len(i_map), max(i_map.keys()))
    entity_total = max(len(e_map), max(e_map.keys()))
    relation_total = max(len(r_map), max(r_map.values()))

    if FLAGS.share_embeddings:
        item_entity_total = len(ikg_map)
        entity_total = item_entity_total
        item_total = item_entity_total

    joint_model = init_model(FLAGS, user_total, item_total, entity_total, relation_total, logger, i_map=i_map, e_map=e_map, new_map=ikg_map)

    triple_epoch_length = math.ceil( float(triple_train_total) / (1-FLAGS.joint_ratio) / FLAGS.batch_size )
    rating_epoch_length = math.ceil( float(rating_train_total) / FLAGS.joint_ratio / FLAGS.batch_size )

    epoch_length = max(triple_epoch_length, rating_epoch_length)
    
    trainer = ModelTrainer(joint_model, logger, epoch_length, FLAGS)

    if FLAGS.load_ckpt_file is not None and FLAGS.share_embeddings:
        load_ckpt_files = FLAGS.load_ckpt_file.split(':')
        for filename in load_ckpt_files:
            trainer.loadEmbedding(os.path.join(FLAGS.log_path, filename), joint_model.state_dict(), e_remap=e_map, i_remap=i_map)
        joint_model.is_pretrained = True
    elif FLAGS.load_ckpt_file is not None:
        load_ckpt_files = FLAGS.load_ckpt_file.split(':')
        for filename in load_ckpt_files:
            trainer.loadEmbedding(os.path.join(FLAGS.log_path, filename), joint_model.state_dict())
        joint_model.is_pretrained = True
    
    # Do an evaluation-only run.
    if only_forward:
        for i, eval_data in enumerate(rating_eval_datasets):
            all_dicts = None
            if FLAGS.filter_wrong_corrupted:
                all_dicts = [rating_train_dict] + [tmp_data[3] for j, tmp_data in enumerate(rating_eval_datasets) if j!=i]
            evaluateRec(
                FLAGS,
                joint_model,
                eval_data[0],
                eval_data[3],
                all_dicts,
                i_map,
                logger,
                eval_descending=True if trainer.model_target == 1 else False,
                is_report=FLAGS.is_report)
        # head_iter, tail_iter, eval_total, eval_list, eval_head_dict, eval_tail_dict
        for i, eval_data in enumerate(triple_eval_datasets):
            all_head_dicts = None
            all_tail_dicts = None
            if FLAGS.filter_wrong_corrupted:
                all_head_dicts = [triple_train_head_dict] + [tmp_data[4] for j, tmp_data in enumerate(triple_eval_datasets) if j!=i]
                all_tail_dicts = [triple_train_tail_dict] + [tmp_data[5] for j, tmp_data in enumerate(triple_eval_datasets) if j!=i]
            evaluateKG(
                FLAGS,
                joint_model,
                eval_data[0],
                eval_data[1],
                eval_data[4],
                eval_data[5],
                all_head_dicts,
                all_tail_dicts,
                e_map,
                logger,
                eval_descending=False,
                is_report=FLAGS.is_report)
    else:
        train_loop(
            FLAGS,
            joint_model,
            trainer,
            rating_train_dataset,
            triple_train_dataset,
            rating_eval_datasets,
            triple_eval_datasets,
            e_map,
            i_map,
            ikg_map,
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
