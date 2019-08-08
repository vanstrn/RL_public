"""
Asynchonous Advantage Actor Critic implementation in Cart-Pole.

"""

from queue import Queue
import numpy as np
# import gym_cap.envs.const as CONST
import gym.spaces

def record(episode,
           episode_reward,
           worker_idx,
           global_ep_reward,
           result_queue,
           total_loss,
           num_steps):
  """Helper function to store score and print statistics.
  Arguments:
    episode: Current episode
    episode_reward: Reward accumulated over the current episode
    worker_idx: Which thread (worker)
    global_ep_reward: The moving average of the global reward
    result_queue: Queue storing the moving average of the scores
    total_loss: The total loss accumualted over the current episode
    num_steps: The number of steps the episode took to complete
  """
  if global_ep_reward == 0:
    global_ep_reward = episode_reward
  else:
    global_ep_reward = global_ep_reward * 0.99 + episode_reward * 0.01
  print(
      f"Episode: {episode} | "
      f"Moving Average Reward: {int(global_ep_reward)} | "
      f"Episode Reward: {int(episode_reward)} | "
      f"Loss: {int(total_loss / float(num_steps) * 1000) / 1000} | "
      f"Steps: {num_steps} | "
      f"Worker: {worker_idx}"
  )
  result_queue.put(global_ep_reward)
  return global_ep_reward

class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []


UNKNOWN = -1
TEAM1_BG = 0
TEAM2_BG = 1
TEAM1_GV = 2
TEAM1_UAV = 3
TEAM2_GV = 4
TEAM2_UAV = 5
TEAM1_FL = 6
TEAM2_FL = 7
OBSTACLE = 8
DEAD = 9
SELECTED = 10
COMPLETED = 11
# UNKNOWN = CONST.UNKNOWN            # -1
# TEAM1_BG = CONST.TEAM1_BACKGROUND  # 0
# TEAM2_BG = CONST.TEAM2_BACKGROUND  # 1
# TEAM1_GV = CONST.TEAM1_UGV         # 2
# TEAM1_UAV = CONST.TEAM1_UAV        # 3
# TEAM2_GV = CONST.TEAM2_UGV         # 4
# TEAM2_UAV = CONST.TEAM2_UAV        # 5
# TEAM1_FL = CONST.TEAM1_FLAG        # 6
# TEAM2_FL = CONST.TEAM2_FLAG        # 7
# OBSTACLE = CONST.OBSTACLE          # 8
# DEAD = CONST.DEAD                  # 9
# SELECTED = CONST.SELECTED          # 10
# COMPLETED = CONST.COMPLETED        # 11
SIX_MAP_CHANNEL = {UNKNOWN: 0, DEAD: 0,
                   TEAM1_BG: 1, TEAM2_BG: 1,
                   TEAM1_GV: 2, TEAM2_GV: 2,
                   TEAM1_UAV: 3, TEAM2_UAV: 3,
                   TEAM1_FL: 4, TEAM2_FL: 4,
                   OBSTACLE: 5}
class fake_agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y
def one_hot_encoder(state, agents=None, vision_radius=19,
                    flatten=False, locs=None):
    """Encoding pipeline for CtF state to one-hot representation
    6-channel one-hot representation of state.
    State is not binary: team2 is represented with -1.
    Channels are not symmetrical.
    :param state: CtF state in raw format
    :param agents: Agent list of CtF environment
    :param vision_radius: Size of the vision range (default=9)`
    :param reverse: Reverse the color. Used for red-perspective (default=False)
    :param flatten: Return flattened representation (for array output)
    :param locs: Provide locations instead of agents. (agents must be None)
    :return oh_state: One-hot encoded state
    """
    if agents is None:
        assert locs is not None
        agents = [fake_agent(x, y) for x, y in locs]

    vision_lx = 2 * vision_radius + 1
    vision_ly = 2 * vision_radius + 1
    oh_state = np.zeros((len(agents), vision_lx, vision_ly, 6), np.float64)

    # team 1 : (1), team 2 : (-1), map elements: (0)
    map_channel = SIX_MAP_CHANNEL
    map_color = {UNKNOWN: 1, DEAD: 0,
                 TEAM1_BG: 0, TEAM2_BG: 1,
                 TEAM1_GV: 1, TEAM2_GV: -1,
                 TEAM1_UAV: 1, TEAM2_UAV: -1,
                 TEAM1_FL: 1, TEAM2_FL: -1,
                 OBSTACLE: 1}

    # Expand the observation with wall to avoid dealing with the boundary
    sx, sy = state.shape
    _state = np.full((sx + 2 * vision_radius, sy + 2 * vision_radius), OBSTACLE)
    _state[vision_radius:vision_radius + sx, vision_radius:vision_radius + sy] = state
    state = _state

    for idx, agent in enumerate(agents):
        # Initialize Variables
        x, y = agent.get_loc()
        x += vision_radius
        y += vision_radius
        vision = state[x - vision_radius:x + vision_radius + 1, y - vision_radius:y + vision_radius + 1]  # extract view

        # FULL MATRIX OPERATION
        for channel, val in map_color.items():
            if val == 1:
                oh_state[idx, :, :, map_channel[channel]] += (vision == channel).astype(np.int32)
            elif val == -1:
                oh_state[idx, :, :, map_channel[channel]] -= (vision == channel).astype(np.int32)

    if flatten:
        return np.reshape(oh_state, (len(agents), -1))
    else:
        return oh_state
