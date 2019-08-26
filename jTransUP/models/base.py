import time
import gflags
import torch
from copy import deepcopy
from functools import reduce
import os

from jTransUP.data import load_rating_data, load_triple_data, load_kg_rating_data

import jTransUP.models.transUP as transup
import jTransUP.models.bprmf as bprmf
import jTransUP.models.transH as transh
import jTransUP.models.jTransUP as jtransup
import jTransUP.models.fm as fm
import jTransUP.models.transE as transe
import jTransUP.models.transR as transr
import jTransUP.models.transD as transd
import jTransUP.models.cofm as cofm
import jTransUP.models.CKE as cke
import jTransUP.models.CFKG as cfkg

def get_flags():
    gflags.DEFINE_enum("model_type", "transup", ["transup", "bprmf", "fm",
                                                "transe", "transh", "transr", "transd",
                                                "cfkg", "cke", "cofm", "jtransup"], "")
    gflags.DEFINE_enum("dataset", "ml1m", ["ml1m", "dbbook2014", "amazon-book", "last-fm", "yelp2018"], "including ratings.csv, r2kg.tsv and a kg dictionary containing kg_hop[0-9].dat")
    gflags.DEFINE_bool(
        "filter_wrong_corrupted",
        True,
        "If set to True, filter test samples from train and validations.")
    gflags.DEFINE_bool("share_embeddings", False, "")
    gflags.DEFINE_bool("use_st_gumbel", False, "")
    gflags.DEFINE_integer("max_queue", 10, ".")
    gflags.DEFINE_integer("num_processes", 4, ".")

    gflags.DEFINE_float("learning_rate", 0.001, "Used in optimizer.")
    gflags.DEFINE_float("norm_lambda", 1.0, "decay of joint model.")
    gflags.DEFINE_float("kg_lambda", 1.0, "decay of kg model.")
    gflags.DEFINE_integer(
        "early_stopping_steps_to_wait",
        70000,
        "How many times will lr decrease? If set to 0, it remains constant.")
    gflags.DEFINE_bool(
        "L1_flag",
        False,
        "If set to True, use L1 distance as dissimilarity; else, use L2.")
    gflags.DEFINE_bool(
        "is_report",
        False,
        "If set to True, use L1 distance as dissimilarity; else, use L2.")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_integer("embedding_size", 64, ".")
    gflags.DEFINE_integer("negtive_samples", 1, ".")
    gflags.DEFINE_integer("batch_size", 512, "Minibatch size.")
    gflags.DEFINE_enum("optimizer_type", "Adagrad", ["Adam", "SGD", "Adagrad", "Rmsprop"], "")
    gflags.DEFINE_float("learning_rate_decay_when_no_progress", 0.5,
                        "Used in optimizer. Decay the LR by this much every epoch steps if a new best has not been set in the last epoch.")

    gflags.DEFINE_integer(
        "eval_interval_steps",
        14000,
        "Evaluate at this interval in each epoch.")
    gflags.DEFINE_integer(
        "training_steps",
        1400000,
        "Stop training after this point.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("margin", 1.0, "Used in margin loss.")
    gflags.DEFINE_float("momentum", 0.9, "The momentum of the optimizer.")
    gflags.DEFINE_integer("seed", 0, "Fix the random seed. Except for 0, which means no setting of random seed.")
    gflags.DEFINE_integer("topn", 10, "")
    gflags.DEFINE_integer("num_preferences", 4, "")
    gflags.DEFINE_float("joint_ratio", 0.5, "(0 - 1). The train ratio of recommendation, kg is 1 - joint_ratio.")

    gflags.DEFINE_string("experiment_name", None, "")
    gflags.DEFINE_string("data_path", None, "")
    gflags.DEFINE_string("rec_test_files", None, "multiple filenames separated by ':'.")
    gflags.DEFINE_string("kg_test_files", None, "multiple filenames separated by ':'.")
    gflags.DEFINE_string("log_path", None, "")
    gflags.DEFINE_enum("log_level", "debug", ["debug", "info"], "")
    gflags.DEFINE_string(
        "ckpt_path", None, "Where to save/load checkpoints. If not set, the same as log_path")
    
    gflags.DEFINE_string(
        "load_ckpt_file", None, "Where to load pretrained checkpoints under log path. multiple filenames separated by ':'.")

    gflags.DEFINE_boolean(
        "has_visualization",
        True,
        "if set True, use visdom for visualization.")
    gflags.DEFINE_integer("visualization_port", 8097, "")
    # todo: only eval when no train.dat when load data
    gflags.DEFINE_boolean(
        "eval_only_mode",
        False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted candidates."
        "Requirements: Must specify load_experiment_name.")
    gflags.DEFINE_string("load_experiment_name", None, "")

def flag_defaults(FLAGS):

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.dataset,
            FLAGS.model_type,
            timestamp,
        )

    if not FLAGS.data_path:
        FLAGS.data_path = "../datasets/"

    if not FLAGS.log_path:
        FLAGS.log_path = "../log/"

    if not FLAGS.ckpt_path:
        FLAGS.ckpt_path = FLAGS.log_path
    
    if FLAGS.seed != 0:
        torch.manual_seed(FLAGS.seed)

    if FLAGS.model_type in ['cke', 'jtransup']:
        FLAGS.share_embeddings = False
    elif FLAGS.model_type == 'cfkg':
        FLAGS.share_embeddings = True


def init_model(
        FLAGS,
        user_total,
        item_total,
        entity_total,
        relation_total,
        logger,
        i_map=None,
        e_map=None,
        new_map=None):
    # Choose model.
    logger.info("Building model.")
    if FLAGS.model_type == "transup":
        build_model = transup.build_model
    elif FLAGS.model_type == "bprmf":
        build_model = bprmf.build_model
    elif FLAGS.model_type == "fm":
        build_model = fm.build_model
    elif FLAGS.model_type == "transe":
        build_model = transe.build_model
    elif FLAGS.model_type == "transh":
        build_model = transh.build_model
    elif FLAGS.model_type == "transr":
        build_model = transr.build_model
    elif FLAGS.model_type == "transd":
        build_model = transd.build_model
    elif FLAGS.model_type == "cofm":
        build_model = cofm.build_model
    elif FLAGS.model_type == "cke":
        build_model = cke.build_model
    elif FLAGS.model_type == "cfkg":
        build_model = cfkg.build_model
    elif FLAGS.model_type == "jtransup":
        build_model = jtransup.build_model
    else:
        raise NotImplementedError

    model = build_model(FLAGS, user_total, item_total, entity_total, relation_total, 
    i_map=i_map, e_map=e_map, new_map=new_map)

    # Print model size.
    logger.info("Architecture: {}".format(model))

    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in model.parameters()])
    logger.info("Total params: {}".format(total_params))

    return model