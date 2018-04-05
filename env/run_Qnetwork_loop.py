# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A run loop for agent/environment interaction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pysc2.neuralnetwork import Qnetwork

import time
import os
import numpy as np
import random
import tensorflow as tf
from pysc2.lib import features
import csv

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


def run_Qnetwork_loop(agents, env, max_frames=0, batch_size=32, update_freq=4 , y=.99, startE=1, endE=0.1,
                      annealing_steps=10000, num_episodes=10000, pre_train_steps=10000, max_epLength=50,
                      load_model=False, path="./dqn", h_size=8, tau=0.001):
  """A run loop to have agents and an environment interact."""
  total_frames = 0
  start_time = time.time()

  # create Qnetworks
  tf.reset_default_graph()
  mainQN = Qnetwork.Qnetwork(h_size)
  targetQN = Qnetwork.Qnetwork(h_size)
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  trainables = tf.trainable_variables()
  targetOps = Qnetwork.updateTargetGraph(trainables, tau)
  myBuffer = Qnetwork.experience_buffer()

  # Set the rate of random action decrease.
  e = startE
  stepDrop = (startE - endE) / annealing_steps

  # create lists to contain total rewards and steps per episode
  jList = []
  rList = []
  meanReward = []
  total_steps = 0

  # Make a path for our model to be saved in.
  if not os.path.exists(path):
    os.makedirs(path)

  action_spec = env.action_spec()
  observation_spec = env.observation_spec()
  for agent in agents:
    agent.setup(observation_spec, action_spec)

  # Create a saver for writing training checkpoints.
  merged_summary_op = tf.summary.merge_all()

  sess = tf.Session()
  sess.run(init)
  result_dir = 'C:/Users/henry_000/Downloads/Pysc2results/4'
  summary_writer = tf.summary.FileWriter(result_dir, sess.graph)  # create summary writer

  i = 0

  try:
    while True:
      episodeBuffer = Qnetwork.experience_buffer()
      timesteps = env.reset()
      i += 1
      for a in agents:
        a.reset()
        s = timesteps[0].observation["screen"][_PLAYER_RELATIVE]
        d = False
        rAll = 0
        j = 0
      while True:
        total_frames += 1
        j += 1
        actions = [agent.step(timestep,e,total_steps,pre_train_steps,sess,mainQN) for agent, timestep in zip(agents, timesteps)]
        if max_frames and total_frames >= max_frames:
          return
        if timesteps[0].last():
          break
        old_score = timesteps[0].observation["score_cumulative"][0]
        timesteps = env.step(actions[0])
        s1 = timesteps[0].observation["screen"][_PLAYER_RELATIVE]
        new_score = timesteps[0].observation["score_cumulative"][0]
        r = 0
        if new_score > old_score:
          r = new_score - old_score
        total_steps += 1
        episodeBuffer.add(np.reshape(np.array([s, actions[0][1], r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.
        if total_steps > pre_train_steps:
          if e > endE:
            e -= stepDrop
          if total_steps % (update_freq) == 0:
            trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
            # Below we perform the Double-DQN update to the target Q-values
            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.stack(trainBatch[:, 3])})
            Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.stack(trainBatch[:, 3])})
            end_multiplier = -(trainBatch[:, 4] - 1)
            doubleQ = Q2[range(batch_size), Q1]
            targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
            # Update the network with our target values.
            summary,_ = sess.run([merged_summary_op,mainQN.updateModel],
                         feed_dict={mainQN.scalarInput: np.stack(trainBatch[:, 0], axis=0), mainQN.targetQ: targetQ,
                                    mainQN.actions: trainBatch[:, 1]})
            Qnetwork.updateTarget(targetOps, sess)  # Update the target network toward the primary network
            # Write logs at every iteration
            mean_100ep_reward = round(np.mean(rList[-101:-1]), 1)
            tf.summary.scalar('Average100Reward', mean_100ep_reward)
            summary_writer.add_summary(summary, i)
        rAll += r
        s = s1
      myBuffer.add(episodeBuffer.buffer)
      jList.append(j)
      rList.append(rAll)
      mean_100ep_reward = round(np.mean(rList[-101:-1]), 1)
      print('Mean 100 reward is {0:.2f}' .format(mean_100ep_reward))
      print('Steps is {:d}'.format(total_steps))
      meanReward.append(mean_100ep_reward)
      # Periodically save the model.
      if total_steps % 10000 == 0:
        saver.save(sess, path + '/model-' + str(i) + '.ckpt')
        print("Saved Model")
  except KeyboardInterrupt:
    pass
  finally:
    resultFile = open("output4.csv", 'w')
    wr = csv.writer(resultFile)
    RewardArray = np.asarray( meanReward[1:-1], dtype=np.float64)
    wr.writerow(meanReward[1:-1])
    resultFile.flush()
    resultFile.close()
    elapsed_time = time.time() - start_time
    print("Took %.3f seconds for %s steps: %.3f fps" % (
        elapsed_time, total_frames, total_frames / elapsed_time))
