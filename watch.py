import numpy as np

from unityagents import UnityEnvironment

from agents.ddpg_agent import Agent

env = UnityEnvironment(file_name='./Tennis_Windows_x86_64/Tennis.exe')

model_name = 'ddpg'

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
states = env_info.vector_observations               # get the current state
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = states.shape[1]

agent = Agent(state_size, action_size, num_agents, seed=0)
agent.actor_local.load_checkpoint(model_name)

score = 0                                          # initialize the score
while True:
    actions = agent.act(states)                      # select an action
    env_info = env.step(actions)[brain_name]        # execute the chosen action
    states = env_info.vector_observations           # update current state
    rewards = env_info.rewards                      # the reward (s, a) => r
    done = env_info.local_done                     # if the episode is done
    score += np.mean(rewards)                       # update the score

    if any(done):
        break

print('Score: {}'.format(score))