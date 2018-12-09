# Continuous Control

### Introduction

This project uses Unity’s Reacher environment which simulates the movement of a double-jointed robotic arm to a target location.  In particular, the 20 agent version is used.  Each agent gets its own copy of the environment.

The environment is defined by a continuous state space with dimension size of 33 corresponding to the position, rotation, velocity and angular velocity of each agent's arm.

The action space is also continuous with a dimension of 4 corresponding the torque of the arm's two joints.  Each torque value has a range from -1 to 1.

The goal of each agent is to keep the arm in the target location for as long as possible.  A reward of 0.1 is given for each time step in which keeping the arm in the right spot is achieved.

The environment is considered solved when the each agent on average garners an average score of at least 30 over a series of 100 episodes.

The code is a modified form of the ddpg-pendulum code provided by Udacity in their Deep Reinforcement Learning Nanodegree Program.

### Getting Started
1. [Download](https://www.python.org/downloads/) and install Python 3.6 or higher if not already installed.
2. Install conda if not already installed.  To install conda, follow these [instructions](https://conda.io/docs/user-guide/install/index.html)
3. Create and activate a new conda environment
```
conda create -n p2_continuous-control python=3.6
conda activate p2_continuous-control
```
3. Clone this GitHub repository
```
git clone https://github.com/lynnolson/p2_continuous-control.git
```
4. Change to the p2_continuous-control directory and install python dependencies by running setup.py
```
cd p2_continuous-control
python setup.py install
```
5. Download the Reacher 20 agent environment from the link below.  Note: The training procedure has only been tested on a Linux with a headless environment (the first option below).
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

6. Place the environment zip in the `p2_continuous-control/` folder, and unzip (or decompress) the file.

### Training Instructions

To train the agent, run train.py

```
python train.py -reacher_env_path Reacher_Linux_NoVis/Reacher.x86_64 -ckpt_path_prefix checkpoint
```

To save a plot of the scores over time (successive episodes), set the argument plot_path to a specific file

```
python train.py -reacher_env_path Reacher_Linux_NoVis/Reacher.x86_64 -ckpt_path_prefix checkpoint -plot_path score_per_episode.png
```
The model weights are saved in two files prefixed by ckpt_path_prefix - one corresponds to actor's network weights and the other to the critic's.  Currently there is no mechanism to recreate the model from these parameters.
When you are done, deactivate the conda environment:
```
conda deactivate
```
### Note
The whole procedure above has only been tested on Ubuntu 16.04.
