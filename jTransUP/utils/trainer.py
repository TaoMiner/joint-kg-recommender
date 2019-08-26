import torch
import torch.optim as optim

import os
from jTransUP.utils.misc import to_gpu, recursively_set_device, USE_CUDA

def get_checkpoint_path(FLAGS, suffix=".ckpt"):
    # Set checkpoint path.
    if FLAGS.ckpt_path.endswith(".ckpt"):
        checkpoint_path = FLAGS.ckpt_path
    else:
        checkpoint_path = os.path.join(FLAGS.ckpt_path, FLAGS.experiment_name + suffix)
    return checkpoint_path

def get_model_target(model_type):
    target = 1 if model_type in ["bprmf", "cofm", "fm"] else -1
    return target

check_rho = 1.0
class ModelTrainer(object):
    def __init__(self, model, logger, epoch_length, FLAGS):
        self.model = model
        self.logger = logger
        self.epoch_length = epoch_length
        self.model_target = get_model_target(FLAGS.model_type)

        self.logger.info('One epoch is ' + str(self.epoch_length) + ' steps.')

        self.parameters = [param for name, param in model.named_parameters()]
        self.optimizer_type = FLAGS.optimizer_type

        self.l2_lambda = FLAGS.l2_lambda
        self.learning_rate_decay_when_no_progress = FLAGS.learning_rate_decay_when_no_progress
        self.momentum = FLAGS.momentum
        self.eval_interval_steps = FLAGS.eval_interval_steps

        self.step = 0
        self.best_step = 0

        # record best dev, test acc
        self.best_dev_performance = 0.0
        self.best_performances = None

        # GPU support.
        to_gpu(model)

        self.optimizer_reset(FLAGS.learning_rate)

        self.checkpoint_path = get_checkpoint_path(FLAGS)

        # Load checkpoint if available.
        if FLAGS.eval_only_mode and os.path.isfile(FLAGS.load_experiment_name):
            self.logger.info("Found checkpoint, restoring.")
            self.load(FLAGS.load_experiment_name, cpu=not USE_CUDA)
            self.logger.info(
                "Resuming at step: {} with best dev performance: {} and test performance : {}.".format(
                    self.best_step, self.best_dev_performance, self.best_performances))
                
    def reset(self):
        self.step = 0
        self.best_step = 0

    def optimizer_reset(self, learning_rate):
        self.learning_rate = learning_rate

        if self.optimizer_type == "Adam":
            self.optimizer = optim.Adam(self.parameters, lr=learning_rate,
                weight_decay=self.l2_lambda)
        elif self.optimizer_type == "SGD":
            self.optimizer = optim.SGD(self.parameters, lr=learning_rate,
                weight_decay=self.l2_lambda, momentum=self.momentum)
        elif self.optimizer_type == "Adagrad":
            self.optimizer = optim.Adagrad(self.parameters, lr=learning_rate,
                weight_decay=self.l2_lambda)
        elif self.optimizer_type == "Rmsprop":
            self.optimizer = optim.RMSprop(self.parameters, lr=learning_rate,
                weight_decay=self.l2_lambda, momentum=self.momentum)

    def optimizer_step(self):
        self.optimizer.step()
        self.step += 1

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def new_performance(self, dev_performance, performances):
        is_best = False
        # Track best dev error
        performance_to_care = dev_performance[0]
        if performance_to_care > check_rho * self.best_dev_performance:
            self.best_step = self.step
            self.logger.info( "Checkpointing ..." )
            self.save(self.checkpoint_path)
            self.best_performances = performances
            self.best_dev_performance = performance_to_care
            is_best = True
        # Learning rate decay
        if self.learning_rate_decay_when_no_progress != 1.0:
            last_epoch_start = self.step - (self.step % self.epoch_length)
            if self.step - last_epoch_start <= self.eval_interval_steps and self.best_step < (last_epoch_start - self.epoch_length):
                    self.logger.info('No improvement after one epoch. Lowering learning rate.')
                    self.optimizer_reset(self.learning_rate * self.learning_rate_decay_when_no_progress)
        return is_best

    def checkpoint(self):
        self.logger.info("Checkpointing.")
        self.save(self.checkpoint_path)

    def save(self, filename):
        if USE_CUDA:
            recursively_set_device(self.model.state_dict(), gpu=-1)
            recursively_set_device(self.optimizer.state_dict(), gpu=-1)

        # Always sends Tensors to CPU.
        save_dict = {
            'step': self.step,
            'best_step': self.best_step,
            'best_dev_performance': self.best_dev_performance,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }
        torch.save(save_dict, filename)

        if USE_CUDA:
            recursively_set_device(self.model.state_dict(), gpu=USE_CUDA)
            recursively_set_device(self.optimizer.state_dict(), gpu=USE_CUDA)

    def load(self, filename, cpu=False):
        if cpu:
            # Load GPU-based checkpoints on CPU
            checkpoint = torch.load(
                filename, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(filename)
        model_state_dict = checkpoint['model_state_dict']

        self.model.load_state_dict(model_state_dict, strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.step = checkpoint['step']
        self.best_step = checkpoint['best_step']
        self.best_dev_performance = checkpoint['best_dev_performance']
    
    def loadEmbedding(self, filename, embedding_names, cpu=False, e_remap=None, i_remap=None):
        assert os.path.isfile(filename), "Checkpoint file not found!"
        self.logger.info("Found checkpoint, restoring pre-trained embeddings.")

        if cpu:
            # Load GPU-based checkpoints on CPU
            checkpoint = torch.load(
                filename, map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(filename)
        old_model_state_dict = checkpoint['model_state_dict']

        model_dict = self.model.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in old_model_state_dict.items() if k in embedding_names}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # for cke load
        if 'ent_embeddings.weight' in old_model_state_dict and 'ent_embeddings.weight' in model_dict and len(old_model_state_dict['ent_embeddings.weight'])+1 == len(self.model.ent_embeddings.weight.data):

            loaded_embeddings = old_model_state_dict['ent_embeddings.weight']
            del (model_dict['ent_embeddings.weight'])
            self.model.ent_embeddings.weight.data[:len(loaded_embeddings), :] = loaded_embeddings[:, :]
            self.logger.info('Restored ' + str(len(loaded_embeddings)) + ' entities from checkpoint.')
        
        # for cfkg load
        if 'rel_embeddings.weight' in old_model_state_dict and 'rel_embeddings.weight' in model_dict and len(old_model_state_dict['rel_embeddings.weight'])+1 == len(self.model.rel_embeddings.weight.data):

            loaded_embeddings = old_model_state_dict['rel_embeddings.weight']
            del (model_dict['rel_embeddings.weight'])
            self.model.rel_embeddings.weight.data[:len(loaded_embeddings), :] = loaded_embeddings[:, :]
            self.logger.info('Restored ' + str(len(loaded_embeddings)) + ' relations from checkpoint.')
        
        # restore entities
        if e_remap is not None and 'ent_embeddings.weight' in model_dict and 'ent_embeddings.weight' in embedding_names:
            loaded_embeddings = model_dict['ent_embeddings.weight']
            del (model_dict['ent_embeddings.weight'])

            count = 0
            for index in e_remap:
                mapped_index = e_remap[index]
                self.model.ent_embeddings.weight.data[mapped_index, :] = loaded_embeddings[index, :]
                count += 1
            self.logger.info('Restored ' + str(count) + ' entities from checkpoint.')
        
        # restore entities
        if i_remap is not None and 'item_embeddings.weight' in model_dict and 'item_embeddings.weight' in embedding_names:
            loaded_embeddings = model_dict['item_embeddings.weight']
            del (model_dict['item_embeddings.weight'])

            count = 0
            for index in i_remap:
                mapped_index = i_remap[index]
                self.model.item_embeddings.weight.data[mapped_index, :] = loaded_embeddings[index, :]
                count += 1
            self.logger.info('Restored ' + str(count) + ' items from checkpoint.')
            # for cofm
            if 'item_bias.weight' in model_dict and 'item_bias.weight' in pretrained_dict:
                loaded_embeddings = model_dict['item_bias.weight']
                del (model_dict['item_bias.weight'])

                count = 0
                for index in i_remap:
                    mapped_index = i_remap[index]
                    self.model.item_bias.weight.data[mapped_index] = loaded_embeddings[index]
                    count += 1
                self.logger.info('Restored ' + str(count) + ' items bias from checkpoint.')

        # 3. load the new state dict
        self.model.load_state_dict(model_dict, strict=False)

        self.logger.info("Load Embeddings of {} from {}.".format(", ".join(list(pretrained_dict.keys())), filename))
