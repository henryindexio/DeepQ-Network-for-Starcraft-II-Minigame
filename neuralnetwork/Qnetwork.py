from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
import random

from pysc2.lib import actions
from pysc2.lib import features

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]
_Feature = features.SCREEN_FEATURES

class Qnetwork():
    def __init__(self, h_size):
        # The network recieves a frame from the game and processes it through two convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None,64,64],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 64, 64, 1])
        self.conv1 = slim.conv2d(
            inputs=self.imageIn, num_outputs=16, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None)
        self.conv2 = slim.conv2d(
            inputs=self.conv1, num_outputs=32, kernel_size=[4, 4], stride=[2, 2], padding='SAME',
            biases_initializer=None)
        self.conv3 = slim.conv2d(
            inputs=self.conv2, num_outputs=h_size, kernel_size=[2, 2], stride=[1, 1], padding='SAME',
            biases_initializer=None)
        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC,self.streamVC = tf.split(self.conv3,2,3)
        self.streamA = tf.reshape(self.streamAC, shape=[-1,(h_size)//2])
        self.streamV = tf.reshape(self.streamVC, shape=[-1,(h_size)//2])
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([(h_size)//2,4]))
        self.VW = tf.Variable(xavier_init([(h_size)//2,1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.AdvantageReshape = tf.reshape(self.Advantage, shape=[-1,64,4])
        self.ValueReshape = tf.reshape(self.Value, shape=[-1,64,1])
        self.AdvantageReduce = tf.reduce_mean(self.AdvantageReshape, axis=1)
        self.ValueReduce = tf.reduce_mean(self.ValueReshape, axis=1)
        self.Qout = self.ValueReduce + tf.subtract(self.AdvantageReduce, tf.reduce_mean(self.AdvantageReduce, axis=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)
        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
        model_variables = slim.get_model_variables()
        variable_summaries(model_variables[0], 'WeightsConv1_')
        variable_summaries(model_variables[1], 'WeightsConv2_')
        variable_summaries(model_variables[2], 'WeightsFC_')

class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return numpy.reshape(numpy.array(random.sample(self.buffer, size)), [size, 5])

def processState(states):
    return numpy.reshape(states, [64*64])


def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars]):
        op_holder.append(tfVars[idx].assign((var.value()*tau) + ((1-tau)*tfVars[idx].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def variable_summaries(var, varname):
    with tf.name_scope(varname+'summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(varname+'mean', mean)
        with tf.name_scope(varname+'stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(varname+'stddev', stddev)
        tf.summary.scalar(varname+'max', tf.reduce_max(var))
        tf.summary.scalar(varname+'min', tf.reduce_min(var))
        tf.summary.histogram(varname+'histogram', var)