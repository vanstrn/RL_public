# RL

File Structure of the code:

- Methods - Methods contain the different Neural Network Methodologies (A3C/PPO/etc) which are able to interface with any supplied network structure.

- Networks - Contains code which creates networks from the provided config files. These are to be generic with layer sizing and depth to be controlled by a config file.

- Utils - Contains utility functions that are used for processing and storing results

- Config - Contains archived configuration files
  - Network - Contatins config files for networks, which contain generic parameters for network structure and parameters
  - Environment - Contains generic config files for different environments.
  - Run - Contains the network and Environment config files which allow a simulation to be run.
  
- Environment - Contains generic functions for handling each environment.

### DL Method
First step is to create a modular neural network structure that is easily able to adapt to scenarios by quick changes to a config file which can be easily archived. This contain explicit names for variable layers as they are used by the methods to determine which layers to train.

The methods are defined to take an input of a network and generic training paramters from a config file. They get specific layer outputs from the network which are used to determine which variables to train and what the appropriate outputs are.

A gerneric run file is used to load a run configuration where it loads or creates a specific network, passes it to a designated method and runs the appropriate environment configuration.
