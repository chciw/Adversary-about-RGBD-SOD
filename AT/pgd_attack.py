"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import nju2000_input
from keras import backend as K
from PIL import Image
import cv2
from tensorflow.python.keras.backend import set_session

sess = tf.Session()
graph = tf.get_default_graph()

class LinfPGDAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start
    self.input_tensors = [
                        model.inputs[0],# input1_0 numpy数组 RGB
                        model.inputs[1],# input2_0 numpy数组 D
                        model.sample_weights[0], # 各个样本的权值，一样就都填 1，是numpy数组
                        model.targets[0], # 输入的标签，是numpy数组
                        K.learning_phase(), # 默认为0，表示test
                    ]
    grad = K.gradients(model.total_loss, model.inputs)
    self.get_gradients = K.function(inputs=self.input_tensors, outputs=grad)
    # if loss_func == 'xent':
    #   loss = model.xent
    # elif loss_func == 'cw':
    #   label_mask = tf.one_hot(model.y_input,
    #                           10,
    #                           on_value=1.0,
    #                           off_value=0.0,
    #                           dtype=tf.float32)
    #   correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
    #   wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
    #   loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    # else:
    #   print('Unknown loss function. Defaulting to cross-entropy')

  def perturb(self, x_nat,d_nat,y):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
      d = d_nat + np.random.uniform(-self.epsilon, self.epsilon, d_nat.shape)
      d = np.clip(d, 0, 255) # ensure valid pixel range
    else:
      x = x_nat.astype(np.float)
      d = d_nat.astype(np.float)

    for i in range(self.num_steps):
      # grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
      #                                       self.model.y_input: y})
      global sess
      global graph
      with graph.as_default():
        set_session(sess)

      grad=self.get_gradients([x,d,np.ones(1), y, 0 ])
      x = np.add(x, self.step_size * np.sign(grad[0]), out=x, casting='unsafe')
      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

      d = np.add(d, self.step_size * np.sign(grad[1]), out=d, casting='unsafe')
      d = np.clip(d, d_nat - self.epsilon, d_nat + self.epsilon)
      d = np.clip(d, 0, 255) # ensure valid pixel range
    return x,d


if __name__ == '__main__':
  import json
  import sys
  import math
  from model import vgg16_deep_fuse_model


  with open('config.json') as config_file:
    config = json.load(config_file)
  # model_file = tf.train.latest_checkpoint(config['model_dir'])
  # if model_file is None:
  #   print('No model found')
  #   sys.exit()

  root = 'D:/PycharmProject/PDNet_available'
  model = vgg16_deep_fuse_model(img_width=224,img_height=224)
  set_session(sess)
  model.load_weights(root + '/checkpoints/vgg16_deep_fuse_512.0.323.hdf5', by_name=True)
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'])
  saver = tf.train.Saver()
  nju2k = nju2000_input.nju2000Data()

  # with tf.Session() as sess:
    # Restore the checkpoint
    # saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
  num_eval_examples = config['num_eval_examples']
  eval_batch_size = config['eval_batch_size']
  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

  x_adv = [] # adv accumulator
  d_adv=[]
  print('Iterating over {} batches'.format(num_batches))

  for ibatch in range(num_batches):
    bstart = ibatch * eval_batch_size
    bend = min(bstart + eval_batch_size, num_eval_examples)
    print('batch {} size: {}'.format(ibatch,bend - bstart))

    x_batch = nju2k.eval_data.xs[bstart:bend, :]
    d_batch = nju2k.eval_data.ds[bstart:bend]
    y_batch = nju2k.eval_data.ys[bstart:bend]
    n_batch = nju2k.eval_data.names[bstart:bend]

    x_batch_adv,d_batch_adv = attack.perturb(x_batch,d_batch,y_batch)
    for i in range(4):
      cv2.imwrite('./advx/'+n_batch[i]+'.png',x_batch_adv[i,:,:,:])
      cv2.imwrite('./advd/' + n_batch[i] + '.png', d_batch_adv[i, :, :, :])
    x_adv.append(x_batch_adv)
    d_adv.append(d_batch_adv)

  print('Storing examples')
  path = config['store_adv_path']
  x_adv = np.concatenate(x_adv, axis=0)
  np.save(path+'pgdx.npy', x_adv)
  d_adv = np.concatenate(d_adv, axis=0)
  np.save(path+'pgdd.npy', d_adv)
  print('Examples stored in {}'.format(path))
