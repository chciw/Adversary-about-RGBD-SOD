"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from model import vgg16_deep_fuse_model
import trainset_input
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])
# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up the data and the model
model = vgg16_deep_fuse_model(224,224)
model.load_weights('../checkpoints/vgg16_deep_fuse_512.0.323.hdf5')

raw_trainset = trainset_input.TrainData()
global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),boundaries,values)
# total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(model.total_loss,global_step=global_step)

# Set up adversary
attack = LinfPGDAttack(model,config['epsilon'],config['num_steps'],config['step_size'],config['random_start'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

'''
会写日志到model_dir下。
cmd cd切换到相应文件夹，tensorboard --logdir=model_dir启动，输入localhost:6006
'''
# 添加变量到直方图中
saver = tf.train.Saver(max_to_keep=3)
# tf.summary.scalar('accuracy adv train', model.accuracy)
# tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('loss adv train', model.total_loss / batch_size)
tf.summary.scalar('loss adv', model.total_loss / batch_size)
tf.summary.image('images adv train', model.inputs[0])
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

with tf.Session() as sess:
  # initialize data augmentation
  trainset = trainset_input.AugmentedTrainData(raw_trainset, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    input_batch,y_batch = trainset.train_data.get_next_batch(batch_size,multiple_passes=True)
    x_batch,d_batch=input_batch[0],input_batch[1]
    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv,d_batch_adv = attack.perturb(x_batch, d_batch,y_batch)
    end = timer()
    training_time += end - start

    nat_dict = {x_batch,d_batch,y_batch}
    adv_dict = {x_batch_adv,d_batch_adv,y_batch}
    # Output to stdout
    if ii % num_output_steps == 0:
      # nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      # adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      # print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      # print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,os.path.join(model_dir, 'checkpoint'),global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start