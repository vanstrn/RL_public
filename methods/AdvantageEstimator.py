
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

def MultiStepDiscountProcessing(reward_list, state_list, gamma:float, steps=40):
    """ MultiStepDiscountProcessing

    Used for calculating multi-step returns:
    G = R_t + γR_t+1 + ... γ^n-1 R_t+n + q(S_t+n,a*,θ-)
    loss = MSE(G,q(S_t,A_t,θ))

    Also associates states for when episodes end early.

    Parameters
    ----------------
    reward_list: list
    value_list: list
    bootstrap: float
    gamma: float
    normalize: boolean (True)

    Returns
    ----------------
    td_target: list
    advantage: list
    """
    rewards = np.append(np.asarray(reward_list).squeeze(),np.zeros(steps))
    value_disc = np.zeros_like(np.asarray(reward_list).squeeze())
    for i in range(steps):
        value_disc = value_disc+gamma**i*rewards[i:len(reward_list)+i]
    if steps  < len(state_list):
        states_n = state_list[steps:]
        for i in range(steps):
            states_n.append(state_list[-1])
    else:
        states_n = [state_list[-1]]*len(state_list)

    return value_disc, states_n


if __name__ == "__main__":

    reward_list = [1,2,3,4,5,6,7,8,9,10]
    gamma = 0.9
    steps = 2
    state_list = [1,2,3,4,5,6,7,8,9,10]
    val,states = MultiStepDiscountProcessing(reward_list,state_list,0,gamma,steps)
    print(states)
