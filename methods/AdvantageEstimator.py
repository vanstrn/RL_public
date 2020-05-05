
import numpy as np
import scipy.signal

def discount_rewards(rewards, gamma, normalized=False, mask_array=None):
    """ take 1D float numpy array of rewards and compute discounted reward

    Args:
        rewards (numpy.array): list of rewards.
        gamma (float): discount rate
        normalize (bool): If true, normalize at the end (default=False)

    Returns:
        numpy.list : Return discounted reward

    """
    if mask_array is None:
        return scipy.signal.lfilter([1.0], [1.0, -gamma], rewards[::-1], axis=-1)[::-1]
    else:
        y, adv = 0.0, []
        mask_reverse = mask_array[::-1]
        for i, reward in enumerate(reversed(rewards)):
            y = reward + gamma * y * (1 - mask_reverse[i])
            adv.append(y)
        disc_r = np.array(adv)[::-1]

        if normalized:
            disc_r = (disc_r - np.mean(disc_r)) / (np.std(disc_r) + 1e-13)

        return disc_r

def gae(reward_list, value_list, bootstrap, gamma:float, lambd:float):
    """ gae

    Generalized Advantage Estimator

    Parameters
    ----------------
    reward_list: list
    value_list: list
    bootstrap: float
    gamma: float
    lambd: float
    normalize: boolean (True)

    Returns
    ----------------
    td_target: list
    advantage: list
    """

    reward_np = np.array(reward_list)
    value_ext = np.array(value_list+[bootstrap])

    td_target  = reward_np + gamma * value_ext[1:]
    advantages = reward_np + gamma * value_ext[1:] - value_ext[:-1]
    advantages = discount_rewards(advantages, gamma*lambd)
    return td_target.tolist(), advantages.tolist()
