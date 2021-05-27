[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/3620840/119905740-a770eb80-bf01-11eb-860e-101970a17098.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Project description

This project uses deep reinforcement learning to teach an agent to play table tennis.

This repo is a fork of the Udacity's deep reinforcement learning [`p3_collab-compet`](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet) repo.

To train a new model, use the `Tennis.ipynb` notebook. In the file `Report.ipybn`, there is an explanation about the algorithm and the training parameters used.

## Environment

The environment consists of two rackets control by the agent playing table tennis against each other. The agent receives a reward of +0.1 every time it hits the ball and a reward of -0.01 if the ball hits the ground or go out of bounds.

![Trained Agent][image1]

### State

The agent's state has 24 dimentions and it is composed of the position and velocity of the ball and racket.

### Actions

The action state is composed of 2 possible actions that are moving the racket close or away from the net and jumping.

### Rewards

The agent receives a reward of +0.1 every time it hits the ball and a reward of -0.01 if the ball hits the ground or go out of bounds.

### Solving the environment

The environment is considered solved if the agent gets an average score of +0.5 over 100 consecutive episodes. Since the agent plays both rackets, these results in two possibly different scores after each episode. We take the maximum over these two possible score and use that as the score of that episode.

## Running the project

### Installing dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__:

    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```

    - __Windows__:

    ```bash
    conda create --name drlnd python=3.6 
    activate drlnd
    ```

2. Install Python dependencies.

```bash
pip install -r requirements.txt
```

3. This project uses Unity to emulate the environment in which the agent takes actions. To run it, you will need to download the environment that matches the operating system you are using.

* Version 1: One agent
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Linux (no visualization): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
* Version 2: Twenty agents
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Linux (no visualization): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

 After downloading the environment, extract the zip file anywhere and update the `Navigation.ipynb` file with the path to the environment folder.

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.

```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

### Trained model

In the `models` folder, you can find the definition of the Neural Networks used for the Actor and Critic. There is only only `model_name` available at this point and it's called `ddpg`.

In the `checkpoints` folder, you can find a trained checkpoint the Actor and Critic.

- checkpoint_{model_name}_actor.pth: a pytorch checkpoint that can be loaded into the actor model.
- checkpoint_{model_name}_critic.pth: a pytorch checkpoint that can be loaded into the critic model.
- scores_{model_name}.pkl: a pickled array containing the scores achieved per episode by the model while training.

To watch a trained model executing actions on the environment use the `watch.py` script.