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
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from pysc2.agents import base_agent
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

class MoveToBeacon_Qnetwork(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  def step(self, obs,e,total_steps,pre_train_steps,sess,mainQN):
    super(MoveToBeacon_Qnetwork, self).step(obs)
    direction_val = 0
    if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      player_y, player_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
      if not player_y.any():
        return [actions.FunctionCall(_NO_OP, []), direction_val]
      player = [int(player_x.mean()), int(player_y.mean())]
      # Choose an action by greedily (with e chance of random action) from the Q-network
      if numpy.random.rand(1) < e or total_steps < pre_train_steps:
          direction_val = numpy.random.randint(4, size=1)
      else:
          direction_val = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [player_relative]})[0]
      direction = ''
      if direction_val == 0:
        direction ='up'
      elif direction_val == 1:
        direction ='down'
      elif direction_val == 2:
        direction = 'left'
      elif direction_val == 3:
        direction = 'right'
      target = move(direction, player)
      #return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, numpy.random.randint(84, size=2)])
      return [actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target]), direction_val]
    else:
      return [actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL]), direction_val]

def move(direction, player):
    player_x = player[0]
    player_y = player[1]
    target = [player_x, player_y]
    if direction == 'up':
        target = [player_x-5, player_y]
    elif direction == 'down':
        target = [player_x+5, player_y]
    elif direction == 'left':
        target = [player_x, player_y-5]
    elif direction == 'right':
        target = [player_x, player_y+5]
    if target[0]<0:
        target[0] = 0
    if target[0]>63:
        target[0] = 63
    if target[1]<0:
        target[1] = 0
    if target[1]>63:
        target[1] = 63

    return target




