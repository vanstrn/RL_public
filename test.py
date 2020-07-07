import numpy as np
import random

reward = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1])
done = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
splits = np.array([True,False,False,False,True,False,False,False,True,False,False,False,True,False,False,True,False,False,False,True,False,False,False,True,False,False,False,True,False,False])
values = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])


split_loc = [i+1 for i, x in enumerate(done) if x]


#Stuff needed for the
reward_lists = np.split(reward,split_loc[:-1])
HL_Critic_lists = np.split(values,split_loc[:-1])
HL_flag_lists = np.split(splits,split_loc[:-1])


for rew,HL_critic,HL_flag in zip(reward_lists,HL_Critic_lists,HL_flag_lists):
    #Colapsing different trajectory lengths for the hierarchical controller
    split_loc_ = [i for i, x in enumerate(HL_flag[:-1]) if x][1:]
    rew_hier = [np.sum(l) for l in np.split(rew,split_loc_)]
    value_hier = [l[0] for l in np.split(HL_critic,split_loc_)]


    #Calculating the td_target and advantage for the hierarchical controller.
    # td_target_i_, advantage_i_ = gae(np.asarray(rew_hier).reshape(-1).tolist(),np.asarray(value_hier).reshape(-1).tolist(),0,self.HPs["Gamma"],self.HPs["lambda"])
    # td_target_hier.extend(td_target_i_); advantage_hier.extend( advantage_i_)

    print(value_hier)
