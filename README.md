
### File Structure of the code:

- Networks - This folder provides network hard-coded network infrastructures. These parameters of the networks can be modified.
The Master branch uses a config file based network creation, ask me for details.

- Methods - Methods contain the different Neural Network Training Methodologies (A3C/PPO/etc). These need to interface with the network.
Right now PPO and A2C are available are set up for simple environment, while the others are set up for slightly different network scheme and require modification

- Utils - Contains utility functions that are used for processing and storing results.

- Configs - Contains archived configuration files
  - Environment - Contains config files for different environments which specify which wrappers to use when creating environment
  - Run - Contains hyperparameters and a pointer to the environmnet files to be used.

- environments - Contains generic functions for handling each environment. I use wrappers around the environments to provide additional functions and to process data into a standard format for the code.

Main execution files are located in root directory.


### Example on Running the code.

`python SingleExec_v2.py -f CTF_PPO.json`

Explanation
`SingleExec_v2.py` is the main execution for a single environment. (AsynchExec.py is an example asynchronous environment execution but is not ported over to this branch.)
`-f CTF_PPO.json` specifies the config file to use (Located in configs/run/) This file specifies execution parameters and hyperparameters for the network.


This will run the an experiment based on the json file.
It will create logs in `log/` and save trained parameters to `model/`

To see results:
1. run `tensorboard --logdir=logs` in a terminal.
2. Open Web browser and go to `http://localhost:6006/`

To run different Network/Method change which methods are loaded in the import calls in SingleExec.
To run different Environments change the EnvConfig in config file.


### Instalation
1. Clone repository

2. Follow instructions here to setup computer for tensorflow GPU.

3. Install Conda and run `conda env create -f rl.yml` to create environment.


###Notes:
- Buffers for training:
I attach a buffer to the Training Class. This is a personal preference as these can be changed based on the method.
The data in the buffer is storred as the following:

buffer = [[tratrajectory 1], [trajectory 2]...] --> This allows for multiple units to be controlled at the same time.
where each trajectory is:
traj # = [
  [s1,s2,..sn,s1,s2,..sn....]  --> List of states from multiple episodes in order
  [a1,a2,..an,a1,a2,..an....]  --> List of states from multiple episodes in order
  [r1,r2,..rn,r1,r2,..rn....]  --> List of states from multiple episodes in order
  [s1,s2,..sn,s1,s2,..sn....]' --> List of next states from multiple episodes in order
  [0, 0,.. 1, 0, 0,.. 1.... ]   --> List defining ends of episodes.
  [....]                       --> Other lists of useful data (Usually in form of network outputs.)
  ]


- Environment setup.
Environments sometimes can be run with standard `gym.make("env_name")` but usually have small differences. Therefore I use wrappers to standardize the environments to the format I want to use. With this I can also add logging functions to the environment.

These work on the principle that the environment is a basic class and that you inherit from it and can add/modify functions. See environmrnts/Commmon.py for some basic wrapper examples.


- Right now code works for Single Agent tasks.
Currently working on getting multiple agents to work on same network.
